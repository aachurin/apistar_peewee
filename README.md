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

For all commands you can specify database using `--database` argument. By default, it's `default`.


### Migrations

Create new migration:
```bash
$ apistar makemigrations
Migration `001_migration_201802051935` has been created.
```

List migrations:
```bash
$ apistar listmigrations
[ ] 001_migration_201802051935
```

Show migration operations:
```bash
$ apistar showmigrations
[ ] 001_migration_201802051935:
  SQL> CREATE TABLE "model1" ("id" SERIAL NOT NULL PRIMARY KEY, "test1" VARCHAR(100) NOT NULL) 
  Python> add_step(step='001_migration_201802051935')
```

Run migrations:
```bash
$ apistar migrate
[X] 001_migration_201802051935
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


- For a working minimal example see [/examples/app.py](https://github.com/aachurin/apistar_peewee/blob/master/examples/app.py).
