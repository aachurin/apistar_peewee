import os
import re
import sys
import peewee
import functools
from datetime import datetime
from apistar_peewee import PeeweeORM
from collections import OrderedDict, namedtuple
from playhouse import migrate as peewee_migrate


INDENT = ' ' * 4
NEWLINE = '\n'
MIGRATE_TEMPLATE = """
{imports}


snapshot = Snapshot()

{snapshot}

def forward(migrator, **kwargs):
    migrator.migrate()


def backward(migrator, **kwargs):
    migrator.migrate()
"""

_DECONSTRUCTORS = {}
_DEFAULT_CALLABLES = {
    datetime.now: ('datetime', 'datetime.now'),
}


Literal = namedtuple('Literal', ['value'])


def deconstruct_field(field, modules=None):
    if modules is None:
        modules = set()

    params = {}
    field_class = type(field)

    default_args = (
        ('null', False),
        ('index', False),
        ('unique', False),
        ('primary_key', False),
        ('constraints', None),
        ('sequence', None),
        ('collation', None),
        ('unindexed', False),
    )

    for attr, value in default_args:
        if getattr(field, attr) != value:
            params[attr] = getattr(field, attr)

    if field.column_name != field.name:
        params['column_name'] = field.column_name

    # Handle extra attributes
    if hasattr(field, 'deconstruct'):
        field.deconstruct(params)
    else:
        for cls in field_class.__mro__:
            if cls in _DECONSTRUCTORS:
                _DECONSTRUCTORS[cls](field, params)
                break

    # Handle default value.
    if field.default is not None:
        if callable(field.default):
            if field.default in _DEFAULT_CALLABLES:
                path = _DEFAULT_CALLABLES[field.default]
            else:
                path = field.default.__module__, field.default.__name__
            modules.add(path[0])
            params['default'] = Literal('%s.%s' % path)
        else:
            params['default'] = field.default

    modules.add(field_class.__module__)
    return params


def field_to_code(field, modules=None):
    params = deconstruct_field(field, modules)
    field_class = type(field)
    param_str = ', '.join('%s=%s' % (k, (v.value if type(v) is Literal else repr(v)))
                            for k, v in sorted(params.items()))
    result = '%s = %s(%s)' % (
        field.name,
        field_class.__module__ + '.' + field_class.__name__,
        param_str)
    return result


def deconstructor(field_class):
    def decorator(fn):
        _DECONSTRUCTORS[field_class] = fn
        return fn
    return decorator


@deconstructor(peewee.DateTimeField)
def datetimefield_deconstructor(field, params):
    if not isinstance(field.formats, list):
        params['formats'] = field.formats


@deconstructor(peewee.CharField)
def charfield_deconstructor(field, params):
    params['max_length'] = field.max_length


@deconstructor(peewee.DecimalField)
def decimalfield_deconstructor(field, params):
    params['max_digits'] = field.max_digits
    params['decimal_places'] = field.decimal_places
    params['auto_round'] = field.auto_round
    params['rounding'] = field.rounding


@deconstructor(peewee.ForeignKeyField)
def deconstruct_foreignkey(field, params):
    default_column_name = field.name
    if not default_column_name.endswith('_id'):
        default_column_name += '_id'
    if default_column_name != field.column_name:
        params['column_name'] = field.column_name
    else:
        params.pop('column_name', None)
    if field.rel_field is not field.rel_model._meta.primary_key:
        params['field'] = field.rel_field.name
    # if field.backref and field.backref != field.model._meta.name + '_set':
    #     params['backref'] = field.backref
    # default_object_id_name = field.column_name
    # if default_object_id_name == field.name:
    #     default_object_id_name += '_id'
    # if default_object_id_name != field.object_id_name:
    #     params['object_id_name'] = field.object_id_name
    if field.on_delete:
        params['on_delete'] = field.on_delete
    if field.on_update:
        params['on_update'] = field.on_update
    if field.model is field.rel_model:
        params['model'] = 'self'
    else:
        params['model'] = Literal('snapshot[%r]' % field.rel_model._meta.name)


class Snapshot:

    def __init__(self, database):
        self.mapping = {}
        self.items = []
        self.database = database

    def __getitem__(self, name):
        return self.mapping[name]

    def __iter__(self):
        return iter(self.items)

    def append(self, model):
        model._meta.database = self.database
        self.items.append(model)
        self.mapping[model._meta.name] = model
        return model


class cached_property:

    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


class Compiler:

    def __init__(self, models, modules=('datetime', 'peewee')):
        self.modules = set(modules)
        self.models = models

    @cached_property
    def snapshot(self):
        models = peewee.sort_models(self.models)
        snapshot = []
        for model in models:
            snapshot.append("@snapshot.append\n" + self.model_to_code(model))
        return snapshot

    @cached_property
    def module_code(self):
        snapshot = NEWLINE + (NEWLINE + NEWLINE).join(self.snapshot)
        imports = NEWLINE.join(['import %s' % x for x in sorted(self.modules)])
        return MIGRATE_TEMPLATE.format(snapshot=snapshot, imports=imports)

    def write(self, path):
        with open(path, 'w') as f:
            f.write(self.module_code)
        return True

    def model_to_code(self, model):
        template = (
            "class {classname}(peewee.Model):\n" +
            INDENT + "{fields}\n" +
            INDENT + "{meta}\n"
        )
        fields = [field_to_code(field, self.modules) for field in model._meta.sorted_fields
                    if not (isinstance(field, peewee.AutoField) and field.name == 'id')]
        meta = ['class Meta:', INDENT + 'table_name = "%s"' % model._meta.table_name]
        if model._meta.schema:
            meta.append(INDENT + 'schema = "%s"')
        # if isinstance(model._meta.primary_key, peewee.CompositeKey):
        #     meta.append(INDENT + 'primary_key = peewee.CompositeKey{0}'.format(model._meta.primary_key.field_names))
        return template.format(
            classname=model.__name__,
            fields=('\n' + INDENT).join(fields),
            meta=('\n' + INDENT).join(meta)
        )


class Router:

    filemask = re.compile(r"[\d]{3}_[^\.]+\.py$")

    def __init__(self,
            database,
            models,
            migrate_table='migratehistory',
            migrate_dir='migrations',
            ):
        self.models = models
        self.database = database
        self.migrate_dir = migrate_dir
        self.migrate_table = migrate_table
        self.migrations = {}

    @cached_property
    def history_model(self):
        """Initialize and cache MigrationHistory model."""
        class MigrateHistory(peewee.Model):
            name = peewee.CharField()
            migrated = peewee.DateTimeField(default=datetime.utcnow)

            class Meta:
                database = self.database
                db_table = self.migrate_table
                # schema = self.schema

            def __str__(self):
                return self.name

        MigrateHistory.create_table(True)
        return MigrateHistory

    @cached_property
    def todo(self):
        """Scan migrations in file system."""
        if not os.path.exists(self.migrate_dir):
            os.makedirs(self.migrate_dir)
        todo = list(sorted(f[:-3] for f in os.listdir(self.migrate_dir) if self.filemask.match(f)))
        return todo

    @cached_property
    def done(self):
        """Scan migrations in database."""
        return [x.name for x in self.history_model.select().order_by(self.history_model.id)]

    @property
    def undone(self):
        """Calculate difference between fs and db."""
        done = set(self.done)
        return [name for name in self.todo if name not in done]

    def get_latest_snapshot(self, migrations):
        todo = list(migrations)
        while todo:
            snapshot = self.read(todo.pop()).get('snapshot')
            if snapshot is not None:
                return snapshot
        return self.read('zero').get('snapshot')

    def exec(self, code):
        scope = {'Snapshot': lambda: Snapshot(self.database)}
        code = compile(code, '<string>', 'exec', dont_inherit=True)
        exec(code, scope)
        return scope

    def read(self, name):
        """Read migration from file."""
        if name not in self.migrations:
            if name == 'zero':
                scope = {'snapshot': Snapshot(self.database)}
            else:
                with open(os.path.join(self.migrate_dir, name + '.py')) as f:
                    scope = self.exec(f.read())
            allowed_attrs = ('snapshot', 'forward', 'backward')
            scope = {k: v for k, v in scope.items() if k in allowed_attrs}
            self.migrations[name] = scope
        return self.migrations[name]

    def create(self, name=None):
        """Create a migration."""
        name = name or datetime.now().strftime('migration_%Y%m%d%H%M')
        name = '{:03}_'.format(len(self.todo) + 1) + name
        compiler = Compiler(self.models)
        if compiler.snapshot == Compiler(self.get_latest_snapshot(self.todo)).snapshot:
            return []
        self.todo.append(name)
        self.migrations[name] = self.exec(compiler.module_code)
        for step in self.migrate(name)[1]:
            step.get_ops()
        path = os.path.join(self.migrate_dir, name + '.py')
        compiler.write(path)
        return [name]

    def migrate(self, migration=None):
        """Run migration."""
        if migration == 'zero':
            forward = False
            migration = None
        else:
            if migration is not None:
                try:
                    prefix = '{:03}_'.format(int(migration))
                    for m in self.todo:
                        if m.startswith(prefix):
                            migration = m
                            break
                except ValueError:
                    pass

            if not self.todo or self.done and migration == self.done[-1]:
                return (True, [])

            migration = migration or self.todo[-1]

            if migration not in self.todo:
                raise KeyError('Unknown migration `%s`' % migration)

            forward = migration not in self.done
        return forward, self.run_forward(migration) if forward else self.run_backward(migration)

    def _forward_backward(self, migrator):
        migrator.migrate()

    def run_forward(self, migration):
        migration_steps = []
        done, undone = self.done, self.undone
        def add_step(step):
            self.history_model.create(name=step)
        while 1:
            step = undone.pop(0)
            curr_snap = self.get_latest_snapshot(done)
            next_snap = self.read(step).get('snapshot')
            forward = self.read(step).get('forward', self._forward_backward)
            migrator = SchemaMigrator.from_database(self.database)
            migrator.setup(step, curr_snap, next_snap)
            forward(migrator)
            migrator.add_op(add_step, step=step)
            migration_steps.append(migrator)
            done.append(step)
            if migration == step:
                break
        return migration_steps

    def run_backward(self, migration):
        migration_steps = []
        done = self.done
        def drop_step(step):
            self.history_model.delete().where(self.history_model.name == step).execute()
        while done:
            step = done[-1]
            if migration == step:
                break
            curr_snap = self.get_latest_snapshot(done)
            next_snap = self.get_latest_snapshot(done[:-1])
            backward = self.read(step).get('backward', self._forward_backward)
            migrator = SchemaMigrator.from_database(self.database)
            migrator.setup(step, curr_snap, next_snap)
            backward(migrator)
            migrator.add_op(drop_step, step=step)
            migration_steps.append(migrator)
            done.pop()
        return migration_steps


def operation(fn):
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        self.add_op(fn(self, *args, **kwargs))
    return inner


Call = namedtuple('Call', ['fn', 'args', 'kwargs'])


class SchemaMigrator:

    def __init__(self, database):
        self.database = database
        self.mops = []
        self.hints = []

    @classmethod
    def from_database(cls, database):
        if isinstance(database, peewee.PostgresqlDatabase):
            return PostgresqlMigrator(database)
        elif isinstance(database, peewee.MySQLDatabase):
            return MySQLMigrator(database)
        else:
            raise NotImplementedError('sqlite is not supported')
            # return SqliteMigrator(database)

    def add_op(self, obj, *args, **kwargs):
        if isinstance(obj, list):
            for o in obj:
                self.add_op(o)
        elif isinstance(obj, peewee.Context):
            self.mops.append(obj)
        elif callable(obj):
            self.mops.append(Call(obj, args, kwargs))
        else:
            raise TypeError('Invalid operation')

    def _alter_table(self, model):
        return model._schema._create_context() \
            .literal('ALTER TABLE ') \
            .sql(model._meta.table)

    def _field_type(self, field):
        ctx = field.model._schema._create_context()
        if ctx.state.field_types:
            return ctx.state.field_types.get(field.field_type, field.field_type)
        return field.field_type

    def _field_ddl(self, field):
        ctx = field.model._schema._create_context()
        return ctx.sql(field.ddl_datatype(ctx)).query()[0]

    def _is_index_for_foreign_key(self, index):
        return len(index._expressions) == 1 and isinstance(index._expressions[0], peewee.ForeignKeyField)

    def get_indexes(self, model):
        result = {}
        for index in model._meta.fields_to_index():
            if self._is_index_for_foreign_key(index):
                continue
            ddl = model._schema._create_context().sql(index).query()[0]
            result[ddl] = index
        return result

    def get_foreign_key_constraints(self, model):
        result = {}
        for field in model._meta.sorted_fields:
            if isinstance(field, peewee.ForeignKeyField):
                ddl = model._schema._create_context().sql(field.foreign_key_constraint()).query()[0]
                result[(ddl, field.unique)] = field
        return result

    def get_foreign_key_name(self, model, field):
        raise NotImplementedError

    def create_table(self, model):
        if self.database.sequences:
            for field in model._meta.sorted_fields:
                if field and field.sequence:
                    self.add_op(model._schema._create_sequence(field))
        self.add_op(model._schema._create_table(safe=False))
        self.add_op(model._schema._create_indexes(safe=False))

    def drop_table(self, model):
        self.add_op(model._schema._drop_table(safe=False))

    def add_index(self, model, index):
        self.add_op(model._schema._create_index(index, safe=False))

    def drop_index(self, model, index):
        self.add_op(model._schema._drop_index(index, safe=False))

    def add_column(self, model, field):
        ctx = self._alter_table(model)
        (ctx.literal(' ADD COLUMN ')
            .sql(field)
            .literal(' ')
            .sql(field.ddl_datatype(ctx)))
        self.add_op(ctx)
        if not field.null:
            if field.default is None:
                raise ValueError('%s.%s is not null but has no default' % (model._meta.name, field.name))
            self.add_not_null(model, field)

    def drop_column(self, model, field):
        ctx = self._alter_table(model)
        (ctx.literal(' DROP COLUMN ')
           .sql(field))
        self.add_op(ctx)

    def alter_column(self, model, field1, field2):
        raise NotImplementedError

    def drop_foreign_key_constraint(self, model, field):
        name = self.get_foreign_key_name(model, field)
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        self.add_op(self._alter_table(model).literal(self.DROP_FOREIGN_KEY).sql(peewee.Entity(name)))
        self.drop_index(model, index)

    def add_foreign_key_constraint(self, model, field):
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        self.add_index(model, index)
        self.add_op(self._alter_table(model).literal(' ADD ').sql(field.foreign_key_constraint()))

    def apply_default(self, model, field):
        default = field.default
        if callable(default):
            default = default()
        if default is not None:
            self.add_op(model._schema._create_context()
                .literal('UPDATE ')
                .sql(model._meta.table)
                .literal(' SET ')
                .sql(peewee.Expression(field.column, peewee.OP.EQ, field.db_value(default), flat=True))
                .literal(' WHERE ')
                .sql(field.column.is_null()))

    def add_not_null(self, model, field):
        self.apply_default(model, field)
        self.add_op(self._add_not_null(model, field))

    def drop_not_null(self, model, field):
        self.add_op(self._drop_not_null(model, field))

    def _add_not_null(self, model, field):
        raise NotImplementedError

    def _drop_not_null(self, model, field):
        raise NotImplementedError


class Migrator:
    """Provide migrations."""

    def setup(self, name, snapshot1, snapshot2):
        self.name = name
        self.snapshot1 = snapshot1
        self.snapshot2 = snapshot2
        self.orm = snapshot1

    def run(self):
        with self.database.transaction():
            for op in self.mops:
                if isinstance(op, peewee.Context):
                    try:
                        self.database.execute(op)
                    except Exception as e:
                        raise Exception(str(e), self.get_op_descr(op))
                else:
                    op.fn(*op.args, **op.kwargs)
        self.mops = []
        self.hints = []

    def get_op_descr(self, obj):
        if isinstance(obj, peewee.Context):
            sql, params = obj.query()
            params = params or ''
            return '%s %s' % (sql, params)
        args = []
        if obj.args:
            args.extend('%r' % x for x in obj.args)
        if obj.kwargs:
            args.extend('%s=%r' % (k, v) for k, v in obj.kwargs.items())
        return '%s(%s)' % (obj.fn.__name__, ', '.join(args))

    def get_ops(self):
        result = []
        for obj in self.mops:
            if isinstance(obj, peewee.Context):
                result.append('SQL> ' + self.get_op_descr(obj))
            else:
                result.append('Python> ' + self.get_op_descr(obj))
        return result

    def migrate(self):
        models1 = peewee.sort_models(self.snapshot1)
        models2 = peewee.sort_models(self.snapshot2)
        models1 = OrderedDict([(m._meta.name, m) for m in models1])
        models2 = OrderedDict([(m._meta.name, m) for m in models2])

        # Add models
        for name in [m for m in models2 if m not in models1]:
            self.create_table(models2[name])

        for name in models1:
            if name in models2:
                self.migrate_model(models1[name], models2[name])

        # Remove models
        for name in [m for m in models1 if m not in models2]:
            self.drop_table(models1[name])

        self.orm = self.snapshot2

    def migrate_model(self, model1, model2):
        """Find difference between Peewee models."""

        indexes1 = self.get_indexes(model1)
        indexes2 = self.get_indexes(model2)

        drop_indexes = [indexes1[key] for key in set(indexes1) - set(indexes2)]
        add_indexes = [indexes2[key] for key in set(indexes2) - set(indexes1)]

        constaints1 = self.get_foreign_key_constraints(model1)
        constaints2 = self.get_foreign_key_constraints(model2)
        drop_constraints = [constaints1[key] for key in set(constaints1) - set(constaints2)]
        add_constraints = [constaints2[key] for key in set(constaints2) - set(constaints1)]

        fields1 = model1._meta.fields
        fields2 = model2._meta.fields
        drop_fields = [fields1[key] for key in set(fields1) - set(fields2)]
        add_fields = [fields2[key] for key in set(fields2) - set(fields1)]
        alter_fields = []

        for key in set(fields1).intersection(fields2):
            field1, field2 = fields1[key], fields2[key]
            if self._field_type(field1) != self._field_type(field2):
                drop_fields.append(field1)
                add_fields.append(field2)
            else:
                alter_fields.append((field1, field2))

        for field in drop_constraints:
            self.drop_foreign_key_constraint(model1, field)

        for index in drop_indexes:
            self.drop_index(model1, index)

        for field in drop_fields:
            self.drop_column(model1, field)

        for field1, field2 in alter_fields:
            self.alter_column(model2, field1, field2)

        for field in add_fields:
            self.add_column(model2, field)

        for index in add_indexes:
            self.add_index(model2, index)

        for field in add_constraints:
            self.add_foreign_key_constraint(model2, field)

    def python(self, func, *args, **kwargs):
        """Run python code."""
        self.add_op(func, *args, **kwargs)


class PostgresqlMigrator(SchemaMigrator, Migrator):

    DROP_FOREIGN_KEY = ' DROP CONSTRAINT '

    def alter_column(self, model, field1, field2):
        if field1.column_name != field2.column_name:
            ctx = self._alter_table(model)
            (ctx.literal(' RENAME COLUMN ')
                .sql(field1)
                .literal(' TO ')
                .sql(field2))
            self.add_op(ctx)
        if self._field_ddl(field1) != self._field_ddl(field2):
            ctx = self._alter_table(model)
            (ctx.literal(' ALTER COLUMN ') \
                .sql(field2) \
                .literal(' TYPE ')
                .sql(field2.ddl_datatype(ctx)))
            self.add_op(ctx)
        if field1.null != field2.null:
            if field2.null:
                self.drop_not_null(model, field2)
            else:
                self.add_not_null(model, field2)

    def _add_not_null(self, model, field):
        return self._alter_table(model).literal(' ALTER COLUMN ').sql(field).literal(' SET NOT NULL')

    def _drop_not_null(self, model, field):
        return self._alter_table(model).literal(' ALTER COLUMN ').sql(field).literal(' DROP NOT NULL')

    def get_foreign_key_name(self, model, field):
        sql = """
            SELECT tc.constraint_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON (tc.constraint_name = kcu.constraint_name AND
                    tc.constraint_schema = kcu.constraint_schema)
            JOIN information_schema.constraint_column_usage AS ccu
                ON (ccu.constraint_name = tc.constraint_name AND
                    ccu.constraint_schema = tc.constraint_schema)
            WHERE
                tc.constraint_type = 'FOREIGN KEY' AND
                tc.table_name = %s AND
                tc.table_schema = %s AND
                ccu.table_name = %s AND
                ccu.table_schema = %s AND
                kcu.column_name = %s AND
                ccu.column_name = %s
        """
        cursor = self.database.execute_sql(sql, (
            model._meta.table_name,
            model._meta.schema or 'public',
            field.rel_model._meta.table_name,
            field.rel_model._meta.schema or 'public',
            field.column_name,
            field.rel_field.column_name
            ))
        return cursor.fetchall()[0][0]


class MySQLMigrator(SchemaMigrator, Migrator):

    DROP_FOREIGN_KEY = ' DROP FOREIGN KEY '

    def alter_column(self, model, field1, field2):
        change_ddl = (field1.column_name != field2.column_name or
                        self._field_ddl(field1) != self._field_ddl(field2))
        if change_ddl:
            ctx = self._alter_table(model)
            (ctx.literal(' CHANGE ')
                .sql(field1)
                .literal(' ')
                .sql(field2)
                .literal(' ')
                .sql(field2.ddl_datatype(ctx)))
            self.add_op(ctx)
            if not field2.null:
                self.add_not_null(model, field2)
        elif field1.null != field2.null:
            if field2.null:
                self.drop_not_null(model, field2)
            else:
                self.add_not_null(model, field2)

    def _add_not_null(self, model, field):
        ctx = self._alter_table(model)
        (ctx.literal(' MODIFY ')
            .sql(field)
            .literal(' ')
            .sql(field.ddl_datatype(ctx))
            .literal(' NOT NULL'))
        return ctx

    def _drop_not_null(self, model, field):
        ctx = self._alter_table(model)
        (ctx.literal(' MODIFY ')
            .sql(field)
            .literal(' ')
            .sql(field.ddl_datatype(ctx)))
        return ctx

    def drop_index(self, model, index):
        ctx = model._schema._drop_index(index, safe=False)
        ctx.literal(' ON ')
        ctx.sql(model._meta.table)
        self.add_op(ctx)

    def get_foreign_key_name(self, model, field):
        sql= """
        SELECT constraint_name
        FROM information_schema.key_column_usage
        WHERE table_name = %s
            AND column_name = %s
            AND table_schema = DATABASE()
            AND referenced_table_name = %s
            AND referenced_column_name = %s"""
        cursor = self.database.execute_sql(sql, (
            model._meta.table_name,
            field.column_name,
            field.rel_model._meta.table_name,
            field.rel_field.column_name
            ))
        return cursor.fetchall()[0][0]
