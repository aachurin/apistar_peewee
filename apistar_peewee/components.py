import contextlib
import importlib
import typing
import warnings
import peewee
from apistar import Component, Settings, exceptions
from apistar.types import ParamAnnotation
from playhouse import pool


class AliasMeta(type):

    def __init__(cls, bases, namespace, kwargs):
        if not hasattr(cls, '_base'):
            cls._base = cls
            cls._cache = {}
            cls._alias = 'default'

    def __getitem__(cls, alias):
        if alias == cls._alias:
            return cls
        try:
            return cls._cache[alias]
        except KeyError:
            pass
        clazz = type('%s[%s]' % (cls._base.__name__, alias), (cls._base,), {'_alias': alias})
        cls._cache[alias] = clazz
        return clazz

    def __hash__(cls, _f=object.__hash__):
        return _f(cls._base)

    def __eq__(cls, other):
        return isinstance(other, type) and issubclass(other, cls._base)


class Session(metaclass=AliasMeta):

    def __init__(self, context, models) -> None:
        self.__dict__ = models
        self._context = context

    @property
    def db(self):
        return self._context.database

    @property
    def transaction(self):
        return self._context.transaction


class Database(metaclass=AliasMeta):
    pass


class PeeweeORM:

    _instance = None  # type: PeeweeORM

    def __init__(self) -> None:
        self._initialized = False
        self._modules = []   # type: typing.List
        self.databases = {}  # type: typing.Dict[str, peewee.Database]
        self.models = {}  # type: typing.Dict[str, typing.Dict[str, peewee.Model]]

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init_orm(self, settings: Settings) -> None:
        assert not self._initialized, "peewee ORM is already initialized"
        config = settings['DATABASE'].copy()
        for alias, database_config in config.items():
            self.init_database(alias, database_config)
        self._initialized = True
        for module in self._modules:
            try:
                importlib.import_module(module.__name__ + '.models')
            except ImportError:
                pass
        del self._modules

    def init_database(self, alias: str, config: typing.Dict) -> None:
        default_migrate_dir = 'migrations' if alias == 'default' else 'migrations_' + alias
        migrate_dir = config.pop('migrate_dir', default_migrate_dir)
        migrate_table = config.pop('migrate_table', 'migratehistory')
        init_module = config.pop('preload_module', 'models')
        engine = config.pop('engine', None)
        database = self.databases.get(alias)

        if database is None or isinstance(database, peewee.Proxy):
            if engine is None:
                raise exceptions.ConfigurationError("peewee database engine is not specified")
            engine_class = getattr(peewee, engine, None)
            if engine_class is None:
                engine_class = getattr(pool, engine, None)
            if not isinstance(engine_class, type) or not issubclass(engine_class, peewee.Database):
                raise exceptions.ConfigurationError("unknown peewee database engine")
            db = engine_class(**config)
            db.migrate_dir = migrate_dir
            db.migrate_table = migrate_table
            if database is None:
                self.databases[alias] = db
            else:
                database.initialize(db)
        else:
            if engine is not None:
                warnings.warn("peewee engine `%s` setting was ignored" % engine)
            database.init(**config)
            database.migrate_dir = migrate_dir
            database.migrate_table = migrate_table

        if init_module:
            try:
                importlib.import_module(init_module)
            except ImportError:
                pass

    def get_database(self, alias: str) -> typing.Union[peewee.Database, peewee.Proxy]:
        try:
            return self.databases[alias]
        except KeyError:
            raise exceptions.ConfigurationError("unknown database alias `%s`" % alias)

    def get_models(self, alias: str) -> typing.Dict[str, peewee.Model]:
        try:
            return self.models[alias]
        except KeyError:
            raise exceptions.ConfigurationError("unknown database alias `%s`" % alias)


def get_model_base(alias: str='default', engine: peewee.Database=None, **options):
    assert alias, "invalid alias"
    orm = PeeweeORM.get_instance()
    db = orm.databases.get(alias)
    if db is None:
        db = peewee.Proxy() if engine is None else engine(None)
        orm.databases[alias] = db
    elif engine is not None:
        warnings.warn("peewee engine `%s` argument was ignored" % engine)
    orm.models.setdefault(alias, {})
    options['database'] = db

    class Model(peewee.Model):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            if cls.__name__ in orm.models[alias]:
                raise exceptions.ConfigurationError(
                    "model `%s` is already registered for alias `%s`" %
                    (cls.__name__, alias))
            orm.models[alias][cls.__name__] = cls
        Meta = type('Meta', (), options)

    return Model


def init_orm(settings: Settings) -> PeeweeORM:
    instance = PeeweeORM.get_instance()
    instance.init_orm(settings)
    return instance


@contextlib.contextmanager
def get_session(orm: PeeweeORM, cls: ParamAnnotation) -> typing.Generator[Session, None, None]:
    with orm.get_database(cls._alias).execution_context() as context:
        yield cls(context, orm.models[cls._alias].copy())


def get_database(orm: PeeweeORM, cls: ParamAnnotation) -> peewee.Database:
    return orm.get_database(cls._alias)


components = [
    Component(PeeweeORM, init=init_orm),
    Component(Session, init=get_session, preload=False),
    Component(Database, init=get_database, preload=False)
]
