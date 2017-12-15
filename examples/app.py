#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Minimum-possible self-contained example of Apistar_Peewee."""


import apistar_peewee
from apistar import Route
from apistar.frameworks.wsgi import WSGIApp as App
from peewee import CharField
from playhouse.shortcuts import model_to_dict


Model = apistar_peewee.get_model_base()

class User(Model):
    name = CharField()


def index(name, session: apistar_peewee.Session):
    user = session.User.create(name=name or __doc__)
    user.save()
    return model_to_dict(user)

routes = (Route('/', 'GET', index), )


settings = {
    'DATABASE': {
        'default': {
            'database': 'test.db',
            'engine': 'SqliteDatabase',
        },
    },
}

app = App(routes=routes,
          settings=settings,
          commands=apistar_peewee.commands,
          components=apistar_peewee.components)


if __name__ in '__main__':
    app.main()
