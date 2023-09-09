import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.GaussianBlur(),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )


def get_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )
