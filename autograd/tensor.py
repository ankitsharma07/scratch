from typing import Callable, List, NamedTuple

import numpy as np


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        depends_on: List[Dependency] = None,
    ) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.shape = data.shape
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor {self.data}, requires_grad={self.requires_grad}"

    def backward(self, grad: "Tensor") -> "Tensor":
        assert self.requires_grad, "Called backward on non-requires-grad Tensor"

        self.grad += grad.data

        for dependency in self.depends_on:
            dependency.tensor.backward()

    def sum(self) -> "Tensor":
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns 0-dim tensor that
    is the sum of all of its elements
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-dim tensor, so each input element
            contributes that much
            """

            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = None

    return Tensor(data, requires_grad, depends_on)
