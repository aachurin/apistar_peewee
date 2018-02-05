import os
import datetime
import unittest
from peewee import *
from apistar_peewee.migrator import SchemaMigrator, Snapshot, field_to_code
from playhouse.reflection import Introspector as BaseIntrospector
from playhouse.reflection import Metadata as BaseMetadata
from playhouse.reflection import PostgresqlMetadata as BasePostgresqlMetadata
from playhouse.reflection import MySQLMetadata as BaseMySQLMetadata


def db_loader(engine, name, db_class=None, **params):
    if db_class is None:
        engine_aliases = {
            SqliteDatabase: ['sqlite', 'sqlite3'],
            MySQLDatabase: ['mysql'],
            PostgresqlDatabase: ['postgres', 'postgresql'],
        }
        engine_map = dict((alias, db) for db, aliases in engine_aliases.items()
                          for alias in aliases)
        if engine.lower() not in engine_map:
            raise Exception('Unsupported engine: %s.' % engine)
        db_class = engine_map[engine.lower()]
    if issubclass(db_class, SqliteDatabase) and not name.endswith('.db'):
        name = '%s.db' % name if name != ':memory:' else name
    return db_class(name, **params)


BACKEND = os.environ.get('BACKEND') or 'sqlite'
PRINT_DEBUG = os.environ.get('PRINT_DEBUG')

IS_SQLITE = BACKEND in ('sqlite', 'sqlite3')
IS_MYSQL = BACKEND == 'mysql'
IS_POSTGRESQL = BACKEND in ('postgres', 'postgresql')


def new_connection():
    return db_loader(BACKEND, 'apistar_peewee_test')


db = new_connection()


class Metadata(BaseMetadata):

    def get_column_type_modifiers(self, table, schema):
        query = """
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %%s AND table_schema = %s""" % schema
        cursor = self.database.execute_sql(query, (table, ))
        result = []
        for name, max_length in cursor.fetchall():
            if max_length is not None:
                result.append((name, {'max_length': max_length}))
        return result

    def get_columns(self, table, schema=None):
        columns = super().get_columns(table, schema)
        def make_field_class(field_class, extra_kwargs):
            def maker(*args, **kwargs):
                kwargs.update(extra_kwargs)
                return field_class(*args, **kwargs)
            return maker
        for name, modifiers in self.get_column_type_modifiers(table, schema):
            columns[name].field_class = make_field_class(columns[name].field_class, modifiers)
        return columns


class PostgresqlMetadata(BasePostgresqlMetadata, Metadata):

    def get_column_type_modifiers(self, table, schema):
        return super().get_column_type_modifiers(table, "'%s'" % (schema or 'public'))


class MySQLMetadata(BaseMySQLMetadata, Metadata):

    def get_column_type_modifiers(self, table, schema):
        return super().get_column_type_modifiers(table, 'DATABASE()')


class Introspector(BaseIntrospector):

    @classmethod
    def from_database(cls, database, schema=None):
        if isinstance(database, PostgresqlDatabase):
            metadata = PostgresqlMetadata(database)
        elif isinstance(database, MySQLDatabase):
            metadata = MySQLMetadata(database)
        else:
            raise NotImplementedError('Not yes implemented')
        # else:
        #     metadata = SqliteMetadata(database)
        return cls(metadata, schema=schema)


class TestModel(Model):
    class Meta:
        database = db


class MigrationTestCase(unittest.TestCase):

    database = db

    def setUp(self):
        if not self.database.is_closed():
            self.database.close()
        self.database.connect()
        self.snapshots = []
        self.all_snapshots = []
        self.get_snapshot()
        self.introspector = Introspector.from_database(self.database)
        super().setUp()

    def tearDown(self):
        super().tearDown()
        for snapshot in self.all_snapshots:
            self.database.drop_tables(list(snapshot))
        self.database.close()

    def get_snapshot(self):
        snapshot = Snapshot(self.database)
        self.snapshots.append(snapshot)
        self.all_snapshots.append(snapshot)
        return snapshot

    def get_migrator(self, name):
        migrator = SchemaMigrator.from_database(self.database)
        snapshot1 = self.snapshots.pop(0)
        migrator.setup(name, snapshot1, self.snapshots[0])
        return migrator

    def run_migrator(self, migrator):
        if PRINT_DEBUG:
            print(migrator.name)
            for op in migrator.get_ops():
                print(op)
        migrator.run()

    def run_migration(self, name):
        assert len(self.snapshots) > 1
        for n in range(0, len(self.snapshots) - 1):
            migrator = self.get_migrator('%s %d' % (name, n + 1))
            migrator.migrate()
            self.run_migrator(migrator)
        if PRINT_DEBUG:
            print()

    def get_models(self):
        return self.introspector.generate_models()

    def assertModelsEqual(self, model1, model2):
        self.assertEqual(model1._meta.table_name, model2._meta.table_name)
        indexes1 = sorted([(tuple(names), unique) for names, unique in model1._meta.indexes])
        indexes2 = sorted([(tuple(names), unique) for names, unique in model2._meta.indexes])
        self.assertEqual(indexes1, indexes2)
        fields1 = {k: field_to_code(v) for k, v in model1._meta.fields.items()}
        fields2 = {k: field_to_code(v) for k, v in model2._meta.fields.items()}
        self.assertEqual(fields1, fields2)


class SchemaMigrationTests(MigrationTestCase):

    def test_add_model(self):
        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test = CharField(max_length=100, unique=True)

        self.run_migration('test_add_model')

        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_drop_model(self):
        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test = CharField(max_length=100, unique=True)

        self.get_snapshot()
        self.run_migration('test_drop_model')

        self.assertFalse('model1' in self.get_models())

    def test_add_field_add_index(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100)

        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test2 = CharField(max_length=200, unique=True, default='')

        self.run_migration('test_add_field_add_index')
        Model1.test2.default = None
        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_indexes(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100, unique=True)
            test2 = CharField(max_length=100)
            class Meta:
                indexes = [
                    (('test1', 'test2'), False)
                ]

        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100, index=True)
            test2 = CharField(max_length=100)
            test3 = CharField(max_length=100, unique=True, null=True)
            class Meta:
                indexes = [
                    (('test2', 'test3'), True)
                ]

        self.run_migration('test_indexes')
        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_drop_field_add_drop_index(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100)
            test2 = CharField(max_length=200, index=True)

        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100, null=True, unique=True)

        self.run_migration('test_drop_field_add_drop_index')
        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_not_null_and_default(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100, null=True)

        self.run_migration('test_not_null_and_default1')
        Model1.create(test1=None)
        Model1.create(test1='test1')

        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=100, default='test2')

        self.run_migration('test_not_null_and_default2')
        data = list(Model1.select().order_by(Model1.test1))
        self.assertEqual(data[0].test1, 'test1')
        self.assertEqual(data[1].test1, 'test2')

        Model1.test1.default = None
        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_alter_type(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        self.run_migration('test_alter_type1')
        Model1.create(test1='a' * 5)

        snapshot = self.get_snapshot()
        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=5)

        self.run_migration('test_alter_type2')
        data = Model1.select().first()
        self.assertEqual(data.test1, 'a' * 5)
        self.assertModelsEqual(Model1, self.get_models()['model1'])

    def test_add_foreign_key(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = IntegerField()

        self.run_migration('test_add_foreign_key1')

        m1 = Model1.create(test1='aaa')
        m2 = Model2.create(test1=m1.id)

        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1)
            test2 = ForeignKeyField(Model1, null=True)

        self.run_migration('test_add_foreign_key2')
        self.assertModelsEqual(Model2, self.get_models()['model2'])

    def test_drop_foreign_key(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1, unique=True)

        self.run_migration('test_drop_foreign_key1')

        m1 = Model1.create(test1='aaa')
        m2 = Model2.create(test1=m1)

        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = IntegerField(unique=True)

        self.run_migration('test_drop_foreign_key2')
        self.assertModelsEqual(Model2, self.get_models()['model2'])

    def test_alter_foreign_key(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1)

        self.run_migration('test_alter_foreign_key1')

        m1 = Model1.create(test1='aaa')
        m2 = Model2.create(test1=m1)

        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1, on_delete='CASCADE')

        self.run_migration('test_alter_foreign_key2')
        Model2.test1.on_delete = None
        self.assertModelsEqual(Model2, self.get_models()['model2'])

    def test_alter_foreign_key_index(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1)

        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        @snapshot.append
        class Model2(TestModel):
            test1 = ForeignKeyField(Model1, unique=True)

        self.run_migration('test_alter_foreign_key_index')
        self.assertModelsEqual(Model2, self.get_models()['model2'])

    def test_data_migration(self):
        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = CharField(max_length=10)

        self.run_migration('test_data_migration1')

        Model1.create(test1='1')
        Model1.create(test1='2')

        snapshot = self.get_snapshot()

        @snapshot.append
        class Model1(TestModel):
            test1 = IntegerField(default=0)

        migrator = self.get_migrator('test_data_migration2')

        data = None

        @migrator.python
        def before():
            nonlocal data;
            data = [(m.id, int(m.test1)) for m in migrator.orm['model1'].select()]

        migrator.migrate()

        @migrator.python
        def after():
            Model1 = migrator.orm['model1']
            for id, value in data:
                q = (Model1.update({Model1.test1: value})
                           .where(Model1.id == id))
                q.execute()

        self.run_migrator(migrator)

        values = [m.test1 for m in Model1.select().order_by(Model1.test1)]
        self.assertEqual(values, [1, 2])

        Model1.test1.default = None
        self.assertModelsEqual(Model1, self.get_models()['model1'])


if __name__ == '__main__':
    unittest.main()
