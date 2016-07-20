# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 13:08:35 2016

@author: heistermann
"""

# adapted with help from 
# http://stackoverflow.com/questions/15023333/simple-tool-library-to-visualize-huge-python-dict
from traits.api import HasTraits, Instance
from traitsui.api import View, VGroup, Item, ValueEditor
from wradlib.io import read_generic_hdf5


def ex_load_hdf5():


    filename = "X:/gpm/level2/2A.GPM.Ku.V620160118.20160327-S004128-E011127.V04A.RT-H5"
    # load rainbow file contents to dict
    rbdict = read_generic_hdf5(filename)#, loaddata=False)

    class DictEditor(HasTraits):
        Object = Instance(object)

        def __init__(self, obj, **traits):
            super(DictEditor, self).__init__(**traits)
            self.Object = obj

        def trait_view(self, name=None, view_elements=None):
            return View(
                VGroup(
                    Item('Object',
                         label='Debug',
                         id='debug',
                         editor=ValueEditor(),  # ValueEditor()
                         style='custom',
                         dock='horizontal',
                         show_label=False), ),
                title='Dictionary Editor',
                width=800,
                height=600,
                resizable=True)

    def dic(my_data):
        b = DictEditor(my_data)
        b.configure_traits()

    dic(rbdict)

# =======================================================
if __name__ == '__main__':
#    ex_load_hdf5()
    filename = "X:/gpm/level2/2A.GPM.Ku.V620160118.20160327-S004128-E011127.V04A.RT-H5"
    # load rainbow file contents to dict
    out = read_generic_hdf5(filename)#, loaddata=False)
    for key in out.keys():
        print key
