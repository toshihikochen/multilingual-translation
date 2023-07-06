from transformers import MT5ForConditionalGeneration, MT5Tokenizer, MT5TokenizerFast


class MT5Model(MT5ForConditionalGeneration):
    def freeze_encoder(self):
        self.shared.requires_grad_ = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.shared.requires_grad_ = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.lm_head.requires_grad_ = False

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.lm_head.requires_grad_ = True


def get_model(pretrained_model_name_or_path, use_fast=True):
    if type(pretrained_model_name_or_path) == list:
        tokenizer_name, model_name = pretrained_model_name_or_path
    else:
        model_name = pretrained_model_name_or_path
        tokenizer_name = pretrained_model_name_or_path

    if use_fast:
        print(f'Using fast tokenizer from {tokenizer_name}')
        tokenizer = MT5TokenizerFast.from_pretrained(tokenizer_name)
    else:
        print(f'Using tokenizer from {tokenizer_name}')
        tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)

    print(f'Loading model from {model_name}')
    model = MT5Model.from_pretrained(model_name)

    return tokenizer, model
