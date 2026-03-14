from monai.losses import DiceCELoss


def build_loss(dice_weight=0.7, ce_weight=0.3):
    return DiceCELoss(
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        lambda_dice=dice_weight,
        lambda_ce=ce_weight,
    )
