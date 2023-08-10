# -*- coding: utf-8 -*-


import inject

from app import dependencies

inject.configure(dependencies.bind, bind_in_runtime=False)
