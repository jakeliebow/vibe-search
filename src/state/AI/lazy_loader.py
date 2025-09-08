class LazyLoader:
    """
    A LazyLoader class to instantiate models only when they are accessed.
    """

    def __init__(self, model_class, *args, _setup=None, **kwargs):
        """
        Initialize the LazyLoader with the model class, its arguments, and an optional setup function.

        :param model_class: The class of the model to be lazily loaded.
        :param args: Positional arguments for the model class.
        :param _setup: An optional setup function to be called after model instantiation.
        :param kwargs: Keyword arguments for the model class.
        """
        self._model_class = model_class
        self._args = args
        self._kwargs = kwargs
        self._setup = _setup
        self._instance = None

    def _load_model(self):
        """
        Instantiate the model if it hasn't been instantiated yet and apply the setup function if provided.
        """
        if self._instance is None:
            self._instance = self._model_class(*self._args, **self._kwargs)
            if self._setup:
                self._instance = self._setup(self._instance)

    def __getattr__(self, name):
        """
        Delegate attribute access to the model instance, loading it if necessary.

        :param name: The attribute name to access.
        :return: The attribute value from the model instance.
        """
        self._load_model()
        return getattr(self._instance, name)

    def __call__(self, *args, **kwargs):
        """
        Allow the LazyLoader to be called like a function, delegating to the model instance.

        :param args: Positional arguments for the model instance.
        :param kwargs: Keyword arguments for the model instance.
        :return: The result of calling the model instance.
        """
        self._load_model()
        return self._instance(*args, **kwargs)
