from . commands import commands
from . components import PeeweeORM
from . components import Session
from . components import Database
from . components import get_model_base
from . components import components
from apistar import exceptions as _exceptions


def apps_load_hook(app, module, **kwargs):
    # for apistar_apps
    if PeeweeORM.get_instance()._initialized:
        raise _exceptions.ConfigurationError("PeeweeORM initialized before application loading complete")
    PeeweeORM.get_instance()._modules.append(module)

