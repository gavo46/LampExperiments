class CharacterMemory:
    def __init__(self):
        self.objects = {}

    def remember_object(self, name, description):
        self.objects[name] = description

    def get_object(self, name):
        return self.objects.get(name)