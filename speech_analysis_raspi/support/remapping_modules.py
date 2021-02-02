import pickle


class ModifiedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # print(module)
        # print(name)
        renamed_module = module
        if module == "backbone.support.configuration_classes":
            renamed_module = "speech_analysis_raspi.support.configuration_classes"
        # print(renamed_module)
        return super(ModifiedUnpickler, self).find_class(renamed_module, name)


def modified_load(file_obj):
    return ModifiedUnpickler(file_obj).load()
