def identity_rewrite(
            model,
            tok,
            records,
            hparams,
            copy=False,
            return_orig_weights=True,
            **kwargs,
        ):
    if return_orig_weights:
        return model, {}
    return model

class IdentityHyperParams:
    def __str__(self) -> str:
        return '"No parameters"'
    
    @staticmethod
    def from_json(path):
        return IdentityHyperParams()