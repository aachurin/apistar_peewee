from . commands import commands # noqa
from . components import PeeweeORM # noqa
from . components import Session # noqa
from . components import Database # noqa
from . components import get_model_base # noqa
from . components import components # noqa
from apistar import exceptions as _exceptions


def apps_load_hook(app, module, **kwargs):
    # for apistar_apps
    if PeeweeORM.get_instance()._initialized:
        raise _exceptions.ConfigurationError("PeeweeORM initialized before application loading complete")
    PeeweeORM.get_instance()._modules.append(module)
