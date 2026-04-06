import numpy as np

class Value:
    """
    Stores a scalar or numpy array and its gradient.
    Builds a computation graph for automatic differentiation (backpropagation).
    """

    def __init__(self, data, _parents=(), _op='', label=''):
        """Initializes a Value object."""
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got {type(data)}")
        
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        data_str = f"array(shape={self.data.shape})" if self.data.ndim > 0 else f"scalar({self.data.item():.4f})"
        grad_str = f"array(shape={self.grad.shape})" if self.grad.ndim > 0 else f"scalar({self.grad.item():.4f})"
        return f"Value(data={data_str}, grad={grad_str}, op='{self._op}')"

    def _unbroadcast(self, grad_in, original_shape):
        """Helper to sum gradient back to an original broadcasted shape."""
        if grad_in.shape == original_shape:
            return grad_in
        
        # 1. Sum over new axes
        ndim_diff = grad_in.ndim - len(original_shape)
        if ndim_diff > 0:
            grad_in = np.sum(grad_in, axis=tuple(range(ndim_diff)))

        # 2. Sum over singleton dimensions
        singleton_dims = tuple(i for i, dim in enumerate(original_shape) if dim == 1)
        if singleton_dims:
            grad_in = np.sum(grad_in, axis=singleton_dims, keepdims=True)

        return grad_in

    def __add__(self, other):
        """Addition operation. Handles broadcasting."""
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Multiplication operation. Handles broadcasting."""
        other = other if isinstance(other, Value) else Value(other)

        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    # --- Commutative operations ---
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # --- Other necessary math operations ---
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __pow__(self, other):
        """Power operation (only supports scalar power)."""
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        
        out_data = self.data ** other
        out = Value(out_data, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    # --- Matrix Operations ---
    @property
    def T(self):
        """Matrix transpose operation."""
        out_data = self.data.T
        out = Value(out_data, (self,), 'transpose')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication (@ operator)."""
        other = other if isinstance(other, Value) else Value(other)

        out_data = self.data @ other.data
        out = Value(out_data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # --- Elementary Functions (exp, log) ---
    def exp(self):
        """Exponential function."""
        clipped_data = np.clip(self.data, -500, 700)
        out_data = np.exp(clipped_data)
        out = Value(out_data, (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm function (log base e)."""
        self._stable_data_for_log = np.maximum(self.data, 1e-15)
        out_data = np.log(self._stable_data_for_log)
        out = Value(out_data, (self,), 'log')

        def _backward():
            self.grad += (1.0 / self._stable_data_for_log) * out.grad

        out._backward = _backward
        return out

    # --- Reduction Operations (sum, mean) ---
    def sum(self, axis=None, keepdims=False):
        """Summation operation."""
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'sum')

        def _backward():
            grad_out = out.grad
            if not keepdims and axis is not None:
                grad_out = np.expand_dims(out.grad, axis)
            
            self.grad += np.ones_like(self.data) * grad_out

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean operation."""
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'mean')

        def _backward():
            if axis is None:
                N = self.data.size
            else:
                N = self.data.shape[axis]
            
            grad_out = out.grad
            if not keepdims and axis is not None:
                grad_out = np.expand_dims(out.grad, axis)
                
            self.grad += (np.ones_like(self.data) * grad_out) / N

        out._backward = _backward
        return out
    
    # --- BACKPROPAGATION ---
    def backward(self):
        """Performs backpropagation starting from this Value node."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            node._backward()
            
        # =================================================================
        # MEMORY MANAGEMENT: BREAKING COMPUTATIONAL GRAPH CLOSURES
        # =================================================================
        # [BEFORE]: 
        # The loop simply ended after calling `node._backward()`. The 
        # computational graph remained fully intact in memory.
        #
        # [THE ISSUE (The 47.7 MB Memory Leak)]: 
        # Inside our math operations (like __mul__ or __add__), the 
        # `_backward` function is a closure that captures the local 
        # variables (specifically the `out` Value object). Because the 
        # `out` object also holds a reference to `_backward`, this creates 
        # a "circular reference". Python's standard reference-counting 
        # garbage collector cannot instantly delete circular references. 
        # Since calculating a 2500x2500 pairwise distance matrix requires 
        # arrays that are 47.7 MB each, doing this ~15 times per iteration 
        # consumed nearly 715 MB of RAM per loop, causing an Out-Of-Memory 
        # (OOM) crash by Iteration 30.
        #
        # [THE FIX]: 
        # By explicitly overwriting `node._backward` with an empty lambda 
        # and clearing the `_prev` parent set, we manually sever the 
        # circular dependencies the exact millisecond backpropagation finishes. 
        # This drops the reference count of those objects to zero, forcing 
        # Python to instantly flush the massive NumPy arrays from RAM.
        # =================================================================
        for node in topo:
            node._backward = lambda: None
            node._prev = set()

    