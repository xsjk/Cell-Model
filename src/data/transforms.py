import torch
from typing import Callable


class RandomFlip:
    def __init__(self, prob: float = 0.5, spatial_dims: tuple[int, ...] = (-3, -2, -1)):
        """
        Parameters
        ----------
        prob : float
            Probability of applying flip for each dimension
        spatial_dims : tuple[int, ...]
            Dimensions to apply flipping (supports negative indexing)
            Default: (-3, -2, -1) for all spatial dimensions
        """
        self.prob = prob
        self.spatial_dims = spatial_dims

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for dim in self.spatial_dims:
            assert dim >= -len(image.shape) and dim < len(image.shape)
            if torch.rand(1) < self.prob:
                image = torch.flip(image, dims=[dim])
                mask = torch.flip(mask, dims=[dim])
        return image, mask


class RandomRotate90:
    def __init__(self, prob: float = 0.5, planes: tuple[tuple[int, int], ...] = ((-2, -1),)):
        """
        Parameters
        ----------
        prob : float
            Probability of applying rotation
        planes : tuple[tuple[int, int], ...]
            Tuple of (dim1, dim2) pairs defining rotation planes
            Default: ((-2, -1),) for XY plane rotation
            For 3D: use ((-3, -2), (-3, -1), (-2, -1)) for all planes
        """
        self.prob = prob
        self.planes = planes

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for plane in self.planes:
            if torch.rand(1) < self.prob:
                k = int(torch.randint(0, 4, (1,)).item())
                image = torch.rot90(image, k=k, dims=plane)
                mask = torch.rot90(mask, k=k, dims=plane)
        return image, mask


class RandomIntensityShift:
    def __init__(self, prob: float = 0.5, shift_range: tuple[float, float] = (-0.1, 0.1)):
        """
        Parameters
        ----------
        prob : float
            Probability of applying intensity shift
        shift_range : tuple[float, float]
            Range of intensity shift values (min, max)
        """
        self.prob = prob
        self.shift_range = shift_range

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.prob:
            shift = torch.empty(1).uniform_(*self.shift_range).item()
            image = torch.clamp(image + shift, 0.0, 1.0)
        return image, mask


class RandomNoise:
    def __init__(self, prob: float = 0.5, noise_std: float = 0.05):
        """
        Parameters
        ----------
        prob : float
            Probability of applying noise
        noise_std : float
            Standard deviation of Gaussian noise
        """
        self.prob = prob
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image, mask


class Compose:
    def __init__(self, transforms: list[Callable]):
        """
        Parameters
        ----------
        transforms : list[Callable]
            List of transformation functions to apply sequentially
        """
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def get_default_3d_transforms(flip_prob: float = 0.5, rotate_prob: float = 0.3, intensity_prob: float = 0.3, noise_prob: float = 0.2) -> Compose:
    """
    Get default 3D data augmentation transforms.

    Parameters
    ----------
    flip_prob : float
        Probability of applying flip transformation
    rotate_prob : float
        Probability of applying rotation transformation
    intensity_prob : float
        Probability of applying intensity shift
    noise_prob : float
        Probability of applying noise

    Returns
    -------
    Compose
        Composition of default 3D transforms
    """
    return Compose([
        RandomFlip(prob=flip_prob, spatial_dims=(-3, -2, -1)),  # Flip in all spatial dims
        RandomRotate90(prob=rotate_prob, planes=((-2, -1),)),  # Rotate in XY plane
        RandomIntensityShift(prob=intensity_prob, shift_range=(-0.1, 0.1)),
        RandomNoise(prob=noise_prob, noise_std=0.05),
    ])


def get_minimal_3d_transforms(flip_prob: float = 0.5, rotate_prob: float = 0.3) -> Compose:
    """
    Get minimal 3D data augmentation transforms.

    Parameters
    ----------
    flip_prob : float
        Probability of applying flip transformation
    rotate_prob : float
        Probability of applying rotation transformation

    Returns
    -------
    Compose
        Composition of minimal 3D transforms
    """
    return Compose([
        RandomFlip(prob=flip_prob, spatial_dims=(-2, -1)),  # Flip in XY plane only
        RandomRotate90(prob=rotate_prob, planes=((-2, -1),)),  # Rotate in XY plane
    ])
