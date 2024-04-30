from PIL import Image

class PILImageIterator():
    def __init__(self, path_list,batch=500):
        self._path_list = path_list
        self._i = 0
        self._batch = batch
        self._last = len(path_list)//batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._path_list):
            raise StopIteration
        value = self._path_list[self._i:self._i + self._batch]

        images = []
        for path in value:
            img = Image.open(path).convert("RGB")
            if img is None:
                print("The path does not exist.")
                assert()
            images.append(img)

        self._i += self._batch
        return images