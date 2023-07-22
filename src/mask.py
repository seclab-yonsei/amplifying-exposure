from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import re

from typing import List


class MaskFillingFunction(object):
    def __init__(
        self,
        tok: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
        device: int,
        do_sample: bool = True,
        min_new_tokens: int = 64,
        max_new_tokens: int = 256,
        no_repeat_ngram_size: int = 3,
        top_p: float = 0.95,
        top_k: int = 40,
        temperature: float = 1.0,
    ):
        super(MaskFillingFunction, self).__init__()

        self.tok = tok
        self.model = model
        self.device = device
        self.do_sample = do_sample
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

        self.pad_token = tok.pad_token
        self.eos_token = tok.eos_token
        self.num_return_sequences = 1

    def __call__(self, masked_texts: List[str]) -> List[str]:
        """Performs a series of processes to fill in the blanks of Masked texts.

        Args:
            masked_texts (List[str]): Masked texts

        Returns:
            List[str]: Complete masked text with mask filled
        """
        ## Fill masks.
        raw_fills = self.replace_masks(masked_texts)
        ## |raw_fills| = (batch_size,)

        ## Extract fills.
        extracted_fills = self.extract_fills(raw_fills)
        ## |extracted_fills| = (batch_size, unknown)

        ## Perturb texts.
        perturbed_texts = self.apply_extracted_fills(
            masked_texts,
            extracted_fills,
        )
        ## |perturbed_texts| = (batch_size,)
        return perturbed_texts

    def count_masks(self, texts: List[str]) -> List[int]:
        """Count the number of masks.

        Args:
            texts (List[str]): A list of texts

        Returns:
            List[int]: A list with the number of masks for each text
        """
        n_masks = [
            len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts
        ]
        ## |n_masks| = (batch_size,)
        return n_masks

    def replace_masks(self, texts: List[str]) -> List[str]:
        """Predict corrupted span with mask filling LM.

        Args:
            texts (List[str]): A list of texts

        Returns:
            List[str]: The result of predicting the mask
        """
        ## Tokenize texts.
        tokens = self.tok(texts, padding=True, return_tensors="pt")
        tokens = tokens.to(self.model.device)

        ## Calculate the maximum number of masks.
        n_expected = self.count_masks(texts)
        stop_id = self.tok.encode(f"<extra_id_{max(n_expected)}>")[0]

        ## Replace each masked span with a sample from T5 mask_model.
        tokens = self.model.generate(
            **tokens,
            do_sample=self.do_sample,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_return_sequences=self.num_return_sequences,
            eos_token_id=stop_id,
        )
        ## |tokens| = (batch_size, 1 + length)

        ## Don't forget detaching from gpu into cpu.
        tokens = tokens.detach().cpu()
        texts = self.tok.batch_decode(tokens, skip_special_tokens=False)
        ## |tokens| = (batch_size, unknown)
        ## |texts| = (batch_size,)
        return texts

    def extract_fills(self, texts: List[str]) -> List[List[str]]:
        """Extract only the words that predicted the mask.

        Args:
            texts (List[str]): The result of predicting the mask

        Returns:
            List[List[str]]: Words that predicted mask
        """
        ## Mask token pattern.
        pattern = re.compile(r"<extra_id_\d+>")

        ## Remove <pad> from beginning of each text.
        texts = [
            x.replace(self.pad_token, "").replace(self.eos_token, "").strip()
            for x in texts
        ]
        ## |texts| = (batch_size,)

        ## Return the text in between each matched mask token.
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]
        ## |extracted_fills| = (batch_size,)

        ## Remove whitespace around each fill.
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        ## |extracted_fills| = (batch_size, unknown)
        return extracted_fills

    def apply_extracted_fills(
        self,
        masked_texts: List[str],
        extracted_fills: List[List[str]],
    ) -> List[str]:
        """Apply each predicted mask to the masked text.

        Args:
            masked_texts (List[str]): Masked texts
            extracted_fills (List[List[str]]): Words that predicted mask

        Returns:
            List[str]: Complete masked text with mask filled
        """
        ## Split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(" ") for x in masked_texts]

        ## Calculate the maximum number of masks.
        n_expected = self.count_masks(masked_texts)

        ## Replace each mask token with the corresponding fill.
        for idx, (text, fills, n) in enumerate(
            zip(tokens, extracted_fills, n_expected)
        ):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        ## join tokens back into text
        texts = [" ".join(x) for x in tokens]
        ## |texts| = (batch_size,)
        return texts
