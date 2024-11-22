from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_minus_epsilon_copy = list(vals)
    vals_minus_epsilon_copy[arg] -= epsilon
    vals_plus_epsilon_copy = list(vals)
    vals_plus_epsilon_copy[arg] += epsilon
    return (f(*vals_plus_epsilon_copy) - f(*vals_minus_epsilon_copy)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add x to the the derivative accumulated on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (no `last_fn` and no `derivative`)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule of the derivatives of this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    set_marked_vars = set()
    sorted_vars = []
    visit_variable(variable, set_marked_vars, sorted_vars)
    # We have the sorted order from left to right, need to reverse for the final topological order
    sorted_vars.reverse()
    return sorted_vars


def visit_variable(
    variable: Variable, vars_set: set, vars_list: List[Variable]
) -> None:
    """Helper function for topological_sort. Creates topological sort for the computation graph without constant variables

    Args:
    ----
        variable: The variable to visit
        vars_set: Set of variables that have been visited
        vars_list: List of variables in topological order

    """
    if variable.unique_id in vars_set:
        return
    if not variable.is_leaf():
        for parent in variable.parents:
            if not parent.is_constant():
                visit_variable(parent, vars_set, vars_list)

    vars_set.add(variable.unique_id)
    vars_list.append(variable)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: Its derivative that we want to propagate backward to the leaves.

        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    sorted_vars = topological_sort(variable)
    # create dict with tuples of each variable and a derivative value

    dict_vars = {var.unique_id: 0.0 for var in sorted_vars}
    dict_vars[variable.unique_id] = deriv

    for var in sorted_vars:
        if var.is_leaf():
            var.accumulate_derivative(dict_vars[var.unique_id])
        else:
            for parent, d_input in var.chain_rule(dict_vars[var.unique_id]):
                if parent.is_constant():
                    continue
                dict_vars[parent.unique_id] += d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values"""
        return self.saved_values
