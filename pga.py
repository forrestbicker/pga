"""
a pga script
"""
# %%
from utils import simplex_projection, entropy_projection

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from functools import lru_cache
from tqdm import trange

# %%


class PGDAttack:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cuda",
        learning_rate: float = 1,
        max_gini: float = 1,
        gini_penalty: float = 0,
        patience: int = 1e9,
        cosine_period: int = 60,
        cosine_decay_mult: float = 0.325,
    ):
        # hyperparams
        self.learning_rate = learning_rate
        self.max_gini = (
            max_gini  # minimum gini index allowed for our entropy projection
        )
        self.gini_penalty = gini_penalty  # continuous penalty on gini
        self.patience = patience  # niter before reinit to prev best
        self.cosine_period = cosine_period  # cosine scheduler param
        self.cosine_decay_mult = cosine_decay_mult  # cosine scheduler param

        # setup the model we're attacking
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.embedding_matrix = self.model.model.embed_tokens.weight.detach().to(device)
        self.unembedding_matrix = self.model.lm_head.weight.detach().to(device).T
        for p in self.model.parameters():
            p.requires_grad = False
            p.grad = None
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def initialize_suffix(self, num_tokens: int = 15, discrete=False) -> torch.Tensor:
        """
        Initializes our learnable relaxed suffix

        Args:
            num_tokens: length of suffix L
            discrete:
                - If True, generates one-hot matrix of random tokens
                - If False, generates uniform random normalized matrix

        Returns:
            random_relaxed_suffix: (L, V) tensor
        """
        L, V = num_tokens, self.tokenizer.vocab_size

        # init to random tokens $\{0, 1\}^{L\cross V}$
        if discrete:
            random_token_ids = torch.randint(
                low=0, high=self.tokenizer.vocab_size, size=(1, L)
            )
            random_one_hot_suffix = torch.zeros((L, V), device=self.device)
            random_one_hot_suffix[torch.arange(
                L), random_token_ids.flatten()] = 1.0
            random_suffix = random_one_hot_suffix

        # init to random weights $[0, 1]^{L\cross V}$
        else:
            rand = torch.rand(size=(L, V), device=self.device)
            random_relaxed_suffix = rand / rand.sum(axis=1, keepdim=True)
            random_suffix = random_relaxed_suffix.requires_grad_(True)

        return random_suffix.requires_grad_(True)

    def forward(
        self, intent: str, relaxed_suffix: torch.Tensor, model_response: str
    ) -> torch.Tensor:
        """
        Runs a forward pass on chat_template(intent, suffix, response).
        Since we use a relaxed suffix, we must directly calculate embeddings

        Returns:
            logits: (L, V) tensor where L = tokens in chat_template(intent, suffix, response)
        """
        # 1. grab embeddings of default chat
        chat_template_embeddings, insertion_ix = self._get_chat_template_embeddings(
            intent, model_response
        )

        # 2. insert our relaxed suffix
        adversarial_suffix_embeddings = (
            relaxed_suffix @ self.embedding_matrix).unsqueeze(0)
        intent_embeddings = chat_template_embeddings[:, :insertion_ix, :]
        assistant_embeddings = chat_template_embeddings[:, insertion_ix:, :]
        combined_embeddings = torch.cat(
            [intent_embeddings, adversarial_suffix_embeddings, assistant_embeddings],
            dim=1,
        )

        # 3. direct forward on embeddings
        return self.model.forward(inputs_embeds=combined_embeddings).logits

    def compute_loss_components(
        self,
        intent: str,
        relaxed_suffix: torch.Tensor,
        target_response: str,
    ) -> torch.Tensor:
        """
        Calculates NLL of target response from model logits

        Returns:
            - relaxed_loss: cross entropy loss
            - avg_gini: average gini index of our relaxed tokens
        """
        # 1. forward pass to get logits
        logits = self.forward(
            intent, relaxed_suffix, target_response
        ).squeeze()

        # 2. grab the response logits
        target_ids = torch.tensor(
            self.tokenizer.encode(target_response), device=self.device
        )[1:]  # discard <bos>
        response_start_ix = self._get_chat_template_embeddings(
            intent)[0].size(1) + relaxed_suffix.size(0) - 1
        response_end_ix = response_start_ix + target_ids.size(0)
        logits_target = logits[response_start_ix:response_end_ix, :]

        # verify our indexing is correct
        # target_tokens = [self.tokenizer.decode(t) for t in target_ids]
        # pred_next_tokens = [self.tokenizer.decode(t) for t in logits_target.argmax(axis=1)]
        # for i in range(len(target_tokens)):
        #     print(target_tokens[:i], pred_next_tokens[i])

        # 3. nll of target
        relaxed_loss = F.cross_entropy(
            logits_target,
            target_ids,
            reduction="sum",
        )

        # NOTE:
        # proposed continuous gini penalty
        frobenius = torch.linalg.matrix_norm(relaxed_suffix, ord="fro")
        L = relaxed_suffix.size(0)
        avg_gini = 1 - (frobenius**2) / L

        return relaxed_loss, avg_gini

    def optimize_attack(
        self, intent: str, target: str, niter: int = 100
    ) -> str:
        """
        Generates attack
        """
        # setup learning loop
        relaxed_suffix = self.initialize_suffix()

        optimizer = torch.optim.Adam(
            [relaxed_suffix], lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_period,
            eta_min=self.cosine_decay_mult * self.learning_rate,
        )

        best_discritized_loss, best_discritized_suffix = 1e9, None
        best_relaxed_loss, best_relaxed_suffix = 1e9, None
        time_since_last_discrete_improvement = 0
        progress_bar = trange(niter, desc="PGD Attack")
        for i in progress_bar:
            # 1. reset gradients
            optimizer.zero_grad()

            # 2. calc loss
            relaxed_loss, avg_gini = self.compute_loss_components(
                intent, relaxed_suffix, target
            )
            loss = relaxed_loss + self.gini_penalty * avg_gini
            loss.backward()
            if relaxed_loss < best_relaxed_loss:
                best_relaxed_loss = relaxed_loss
                best_relaxed_suffix = relaxed_suffix.clone().detach()

            # 3. backprop
            pre_update = relaxed_suffix.clone().detach()
            torch.nn.utils.clip_grad_norm_([relaxed_suffix], max_norm=20.0)
            optimizer.step()
            post_update = relaxed_suffix.clone().detach()
            scheduler.step()

            with torch.no_grad():
                # 4. apply projections in place

                # dynamic entropy projection
                lr = optimizer.param_groups[0]['lr']
                # loss_diff = (relaxed_loss - discritized_loss).clamp(min=0)
                # dynamic_max_gini = self.max_gini * (lr / self.learning_rate) * (1 + loss_diff)

                relaxed_suffix[:] = simplex_projection(
                    relaxed_suffix
                )
                relaxed_suffix[:] = entropy_projection(
                    relaxed_suffix, max_gini=self.max_gini
                )

                # 5. calc discritized loss
                most_probable_tokens = relaxed_suffix.argmax(axis=-1)
                discritized_ids = most_probable_tokens
                discritized_ids = self.tokenizer.encode(
                    self.tokenizer.decode(most_probable_tokens), return_tensors="pt"
                )[:, 1:].to(self.device)

                # TODO:
                # proposed improved decoding schema to reduce discritization error
                # instead of using the most probable tokens, use the tokens that best approximate
                # our relaxed_suffix's embeddings per our unembedding matrix
                # adversarial_suffix_embeddings = relaxed_suffix @ self.embedding_matrix
                # discritized_ids = (adversarial_suffix_embeddings @ self.unembedding_matrix).argmax(axis=1)

                discritized_suffix = F.one_hot(
                    discritized_ids, num_classes=relaxed_suffix.size(-1)
                ).float()
                discritized_loss, _ = self.compute_loss_components(
                    intent, discritized_suffix, target
                )

            # 6. patience
            if discritized_loss < best_discritized_loss:
                best_discritized_loss = discritized_loss
                best_discritized_suffix = discritized_suffix
                time_since_last_discrete_improvement = 0
            elif time_since_last_discrete_improvement > self.patience:
                with torch.no_grad():
                    relaxed_suffix[:] = best_discritized_suffix
                time_since_last_discrete_improvement = 0
            time_since_last_discrete_improvement += 1

            # 7. log
            p_target = torch.exp(-relaxed_loss)
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "p_target": f"{p_target:.4f}",
                    "avg_gini": f"{avg_gini:.4f}",
                    "lr": f"{lr:.4f}",
                    "t": f"{time_since_last_discrete_improvement}",
                    "d_loss": f"{discritized_loss:.4f}",
                    "d_loss*": f"{best_discritized_loss:.4f}",
                    "delta norm": f"{post_update.norm().item() - pre_update.norm().item():.4f}",
                }
            )

            if i % 20 == 0:
                print(self.get_model_response(intent, best_relxed_suffix))

            if p_target > 0.99:
                break

        # adversarial_suffix_str = self.tokenizer.decode(
        #     best_discritized_suffix.argmax(axis=-1)
        # )
        niters_complete = i
        return best_relaxed_suffix, best_discritized_suffix, niters_complete

    def get_model_response(self, prompt: str, relaxed_suffix: torch.Tensor) -> str:
        """
        Generates most probable model response to a given prompt
        """
        # 1. grab embeddings of default chat
        chat_template_embeddings, insertion_ix = self._get_chat_template_embeddings(
            prompt)

        # 2. insert our relaxed suffix
        adversarial_suffix_embeddings = (
            relaxed_suffix @ self.embedding_matrix).unsqueeze(0)
        prompt_embeddings = chat_template_embeddings[:, :insertion_ix, :]
        assistant_marker_embeddings = chat_template_embeddings[:,
                                                               insertion_ix:, :]
        combined_embeddings = torch.cat(
            [prompt_embeddings, adversarial_suffix_embeddings,
                assistant_marker_embeddings],
            dim=1,
        )

        # 3. prompt model
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            ).squeeze()

        # 4. decode generated tokens
        gen_text = self.tokenizer.decode(
            generated_ids, clean_up_tokenization_spaces=True
        )
        return gen_text

    @lru_cache
    def _get_chat_template_embeddings(
        self, prompt: str, model_response: str = None
    ) -> tuple[str]:
        # generate fixed template embeddings
        chat_template = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if model_response is not None:
            chat_template.append(
                {"role": "assistant", "content": model_response})
        chat_ids = self.tokenizer.apply_chat_template(
            chat_template,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=model_response is None,
        ).to(self.device)
        chat_template_embeddings = self.model.model.embed_tokens(chat_ids)

        # 2. ix to splice in our suffix
        insertion_ix = self._get_insertion_ix(prompt)
        return chat_template_embeddings, insertion_ix

    def _get_insertion_ix(self, prompt: str):
        # figure out where user content ends
        user_only_template = [{"role": "user", "content": prompt}]
        user_ids = self.tokenizer.apply_chat_template(
            user_only_template,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(self.device)
        user_len = user_ids.size(1)
        return user_len
