## Apistar Peewee ORM


### Install

To use this you first need to install `apistar_peewee` and your chosen database driver:

```bash
$ pip install apistar_peewee
$ pip install psycopg2
```

For Sqlite you do not need any database driver.


### Settings

You then need to add the database config to your settings:

```python
from apistar.frameworks.wsgi import WSGIApp as App
import apistar_peewee


routes = [
   # ...
]

# Configure database settings.
settings = {
    'DATABASE': {
        'default': {
            'database': 'db1',
            'host': '127.0.0.1',
            'port': 5433,
            'user': 'user',
            'password': 'password',
            'engine': 'PooledPostgresqlDatabase'
        }
    }
}


app = App(
    routes=routes,
    settings=settings,
    commands=apistar_peewee.commands,  # Install custom commands.
    components=apistar_peewee.components  # Install custom components.
)
```

You can configure as many databases as you need:

```python
settings = {
    'DATABASE': {
        'default': {
            'database': 'db1',
            'host': '127.0.0.1',
            'port': 5433,
            'user': 'user',
            'password': 'password',
            'engine': 'PooledPostgresqlDatabase'
        },
        'another_db': {
            'database': 'db2',
            'host': '127.0.0.1',
            'port': 5433,
            'user': 'user',
            'password': 'password',
            'engine': 'PooledPostgresqlDatabase'
        }
    }
}
```

Pass arguments to the database engine/driver:

```python
settings = {
    'DATABASE': {
        'default': {
            'database': 'db1',
            'engine': 'PooledPostgresqlExtDatabase',
            'fields': {'varchar': 'varchar'},  # Extra arguments for database driver.
            'register_hstore': false
        }
    }
}

```

The apistar_peewee config above is similar to this Peewee config:

```python
db = PooledPostgresqlExtDatabase("db1",
                                 fields={'varchar': 'varchar'},
                                 register_hstore=False)

```


### Models

Create a new model:

```python
from apistar_peewee import get_model_base
from peewee import *

Model = get_model_base()

class Customer(Model):
    name = CharField(max_length=255)

```

And for all other databases:

```python

AnotherModel = get_model_base(alias='another_db')

class AnotherCustomer(AnotherModel):
    name = CharField(max_length=255)

```

**Creating the database tables:**
=======
All models which name starts with `_` or `Abstract` are abstract models.
Such models are ignored by `createtables` and `makemigrations` commands.

Before starting your app you will likely need to create the database tables which you can do with the following command:

```bash
$ apistar createtables
```

Or use migrations.

For all commands you can specify database using `--database` argument. By default, it's `default`.


### Migrations

Previous model state:
```python
class Tweet(peewee.Model):
    f1 = peewee.CharField(max_length=50)
    price = peewee.IntegerField()
    price2 = peewee.IntegerField()
```

New model state:
```python
class Tweet(peewee.Model):
    f1 = CharField(max_length=20)
    price = IntegerField()
    price2 = CharField(max_length=200)
    price3 = CharField(max_length=200)
```

Create new migration:
```bash
$ apistar makemigrations
Migration `0002_migration_201802142308` has been created.
Line 25: Value '' is selected as the default value for field tweet.price3 since the field is not null
Line 27: Check the field tweet.price2 are correctly converted to string
Line 36: Check the field tweet.price2 are correctly converted to integer
```

List migrations:
```bash
$ apistar listmigrations
[X] 0001_migration_201802142130
[ ] 0002_migration_201802142308
```

Show migration operations:
```bash
$ apistar showmigrations
[ ] 0002_migration_201802142308:
  SQL> ALTER TABLE "tweet" ADD COLUMN "price3" VARCHAR(200) []
  SQL> ALTER TABLE "tweet" RENAME COLUMN "f1" TO "old__f1" []
  SQL> ALTER TABLE "tweet" ADD COLUMN "f1" VARCHAR(20) []
  SQL> ALTER TABLE "tweet" RENAME COLUMN "price2" TO "old__price2" []
  SQL> ALTER TABLE "tweet" ADD COLUMN "price2" VARCHAR(200) []
  SQL> UPDATE "tweet" SET "price3" = %s WHERE ("price3" IS %s) ['', None]
  SQL> UPDATE "tweet" SET "price2" = CAST("old__price2" AS VARCHAR) WHERE ("old__price2" IS NOT %s) [None]
  SQL> UPDATE "tweet" SET "f1" = SUBSTRING("old__f1", %s, %s) WHERE ("old__f1" IS NOT %s) [1, 20, None]
  SQL> ALTER TABLE "tweet" DROP COLUMN "old__f1" []
  SQL> ALTER TABLE "tweet" DROP COLUMN "old__price2" []
  SQL> ALTER TABLE "tweet" ALTER COLUMN "price3" SET NOT NULL []
  SQL> ALTER TABLE "tweet" ALTER COLUMN "f1" SET NOT NULL []
  SQL> ALTER TABLE "tweet" ALTER COLUMN "price2" SET NOT NULL []
  PY>  set_done('0002_migration_201802142308')
```

Run migrations:
```bash
$ apistar migrate
[X] 0002_migration_201802142308

$ apistar listmigrations
[X] 0001_migration_201802142130
[X] 0002_migration_201802142308
```

Rollback migrations:
```
$ apistar showmigrations --to 1
[X] 0002_migration_201802142308:
  SQL> ALTER TABLE "tweet" RENAME COLUMN "price2" TO "old__price2" []
  SQL> ALTER TABLE "tweet" ADD COLUMN "price2" INTEGER []
  SQL> ALTER TABLE "tweet" RENAME COLUMN "f1" TO "old__f1" []
  SQL> ALTER TABLE "tweet" ADD COLUMN "f1" VARCHAR(50) []
  SQL> UPDATE "tweet" SET "price2" = CAST("old__price2" AS INTEGER) WHERE ("old__price2" IS NOT %s) [None]
  SQL> UPDATE "tweet" SET "f1" = SUBSTRING("old__f1", %s, %s) WHERE ("old__f1" IS NOT %s) [1, 50, None]
  SQL> ALTER TABLE "tweet" DROP COLUMN "price3" []
  SQL> ALTER TABLE "tweet" DROP COLUMN "old__price2" []
  SQL> ALTER TABLE "tweet" DROP COLUMN "old__f1" []
  SQL> ALTER TABLE "tweet" ALTER COLUMN "price2" SET NOT NULL []
  SQL> ALTER TABLE "tweet" ALTER COLUMN "f1" SET NOT NULL []
  PY>  set_undone('0002_migration_201802142308')

$ apistar migrate --to 1
[ ] 0002_migration_201802142308

$ apistar listmigrations
[X] 0001_migration_201802142130
[ ] 0002_migration_201802142308

$ apistar showmigrations --to zero
[X] 0001_migration_201802142130:
  SQL> DROP TABLE "tweet" []
  PY>  set_undone('0001_migration_201802142130')
```

To control where the migrations are created,
you can specify `migrate_dir` and `migrate_table` settings (for each database).
Default values are `migrations` and `migratehistory`.


### Database use

To interact with the database, use the `Session` component.
This will automatically handle commit/rollback behavior,
depending on if the view returns normally, or raises an exception:

```python
from apistar_peewee import Session


def create_customer(session: Session, name: str):
    customer = session.Customer(name=name)
    customer.save()
    return {'id': customer.id, 'name': customer.name}


def list_customers(session: Session):
    customers = session.Customer.select()
    return [
        {'id': customer.id, 'name': customer.name}
        for customer in customers
    ]

# and using another_db
def list_another_customers(session: Session['another_db']):
    customers = session.AnotherCustomer.select()
    return [
        {'id': customer.id, 'name': customer.name}
        for customer in customers
    ]

```


In addition, you can use Database instance directly:

```python
from apistar_peewee import Database


def create_another_customers(db: Database['another_db'], name: str):
    with db.execution_context():
        customer = AnotherCustomer.create(name=name)
    return {'id': customer.id, 'name': customer.name}

```


For a working minimal example see [/examples/app.py](https://github.com/aachurin/apistar_peewee/blob/master/examples/app.py).
