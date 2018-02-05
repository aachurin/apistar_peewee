import sys
from apistar import Command, exceptions
from apistar.interfaces import Console, App
from . components import PeeweeORM


color_names = ('black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')


class Colorizer:

    foreground = {color_names[x]: '3%s' % x for x in range(8)}
    background = {color_names[x]: '4%s' % x for x in range(8)}
    opt_dict = {'bold': '1', 'underscore': '4', 'blink': '5', 'reverse': '7', 'conceal': '8'}

    def __init__(self):
        self.supports_color = self.supports_color()

    def supports_color(self):
        """
        Return True if the running system's terminal supports color,
        and False otherwise.
        """
        plat = sys.platform
        supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
        # isatty is not always implemented, #6223.
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        return supported_platform and is_a_tty

    def colored(self, text, fg=None, bg=None):
        if self.supports_color:
            code_list = []
            if fg is not None:
                code_list.append(self.foreground[fg])
            if bg is not None:
                code_list.append(self.background[bg])
            if code_list:
                text = ('\x1b[%sm' % ';'.join(code_list)) + text + '\x1b[0m'
        return text


def get_models(models):
    result = []
    for model in models.values():
        # special case for abstract models
        if (model.__name__.startswith('_') or
            model.__name__.startswith('Abstract') or
            getattr(model._meta, 'nonmigratable', False) or
            getattr(model, '_nonmigratable_', False)):
            continue
        result.append(model)
    return result


def createtables(orm: PeeweeORM, database: str='default'):
    """Create non-abstract tables"""
    models = get_models(orm.get_models(database))
    orm.get_database(database).create_tables(models, safe=True)


def droptables(orm: PeeweeORM, database: str='default'):
    """Drop all tables"""
    models = get_models(orm.get_models(database))
    orm.get_database(database).drop_tables(models.values(), safe=True)


def makemigrations(console: Console, orm: PeeweeORM, name: str='', database: str='default'):
    """Create migration"""
    from . migrator import Router
    c = Colorizer()
    router = Router(database=orm.get_database(database), models=get_models(orm.get_models(database)))
    result = router.create()
    if not result:
        console.echo(c.colored('No changes found.', 'green'))
    else:
        console.echo(c.colored('Migration `%s` has been created.' % result[0], 'yellow'))


def migrate(console: Console, orm: PeeweeORM, migration: str='', database: str='default'):
    """Run migrations"""
    from . migrator import Router
    c = Colorizer()
    router = Router(database=orm.get_database(database), models=get_models(orm.get_models(database)))
    try:
        forward, steps = router.migrate(migration)
    except KeyError as e:
        console.echo(c.colored('; '.join(e.args), 'red'))
        return
    if not steps:
        console.echo(c.colored('There is nothing to migrate', 'yellow'))
        return
    for step in steps:
        step.run()
        if forward:
            console.echo(c.colored('[X] ' + step.name, 'green'))
        else:
            console.echo(c.colored('[ ] ' + step.name, 'yellow'))


def listmigrations(console: Console, orm: PeeweeORM, database: str='default'):
    """List migrations"""
    from . migrator import Router
    c = Colorizer()
    router = Router(database=orm.get_database(database), models=get_models(orm.get_models(database)))
    done = router.done
    for name in done:
        console.echo(c.colored('[X] ' + name, 'green'))
    for name in router.undone:
        console.echo(c.colored('[ ] ' + name, 'yellow'))


def showmigrations(console: Console, orm: PeeweeORM, migration: str='', database: str='default'):
    """Show migrations instructions"""
    from . migrator import Router
    c = Colorizer()
    router = Router(database=orm.get_database(database), models=get_models(orm.get_models(database)))
    try:
        forward, steps = router.migrate(migration)
    except KeyError as e:
        console.echo(c.colored('; '.join(e.args), 'red'))
        return
    if not steps:
        console.echo(c.colored('There is nothing to migrate', 'yellow'))
        return
    for step in steps:
        if forward:
            console.echo(c.colored('[ ] ' + step.name + ':', 'yellow'))
        else:
            console.echo(c.colored('[X] ' + step.name + ':', 'green'))
        for op in step.get_ops():
            console.echo(c.colored('  ' + op, 'cyan'))
        console.echo()


def listmodels(console: Console, orm: PeeweeORM, database: str='default'):
    """List all models"""
    models = get_models(orm.get_models(database))
    for model in models:
        console.echo(model.__name__)


commands = [
    Command('createtables', createtables),
    Command('droptables', droptables),
    Command('migrate', migrate),
    Command('makemigrations', makemigrations),
    Command('listmigrations', listmigrations),
    Command('showmigrations', showmigrations),
    Command('listmodels', listmodels)
]
