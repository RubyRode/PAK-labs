class Item:
    def __init__(self, count=3, max_count=16, index: int = None):
        self._count = count
        self._max_count = 16
        self._index = index

    def __str__(self):
        return f"Item: {self._count}, {self._max_count}"

    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        return False

    def update_index(self, index = None):
        self._index = index

    @property
    def index(self):
        return self._index

    def __add__(self, num):
        """ Сложение с числом """
        if self._max_count < self._count + num < 0:
            raise Exception("Max count exceeded")
        self._count += num
        return self

    def __mul__(self, num):
        """ Умножение на число """
        if self._max_count < self._count * num < 0:
            raise Exception("Max count exceeded")
        self._count *= num
        return self

    def __sub__(self, num):
        """ Вычитание числа """
        if self._max_count < self._count - num < 0:
            raise Exception("Max count exceeded")
        self._count -= num
        return self

    def __iadd__(self, num):
        """ Сложение с числом и присваиванием"""
        if self._max_count < self._count + num < 0:
            raise Exception("Max count exceeded")
        self._count += num
        return self

    def __isub__(self, num):
        """ Вычитание числа """
        if self._max_count < self._count - num < 0:
            raise Exception("Max count exceeded")
        self._count -= num
        return self

    def __imul__(self, num):
        """ Умножение на число """
        if self._max_count < self._count * num < 0:
            raise Exception("Max count exceeded")
        self._count *= num
        return self

    def __gt__(self, num):
        """ Сравнение больше """
        return self._count > num

    def __lt__(self, num):
        """ Сравнение меньше """
        return self._count < num

    def __le__(self, num):
        """ Сравнение меньше или равно """
        return self._count <= num

    def __ge__(self, num):
        """ Сравнение меньше или равно """
        return self._count >= num

    def __eq__(self, num):
        """ Сравнение равно """
        return self._count == num

    def __len__(self):
        """ Получение длины объекта """
        return self._count

    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return self._count


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe

    def __str__(self):
        return f"Fruit: {self._ripe}"

class Vegetable(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe

    def __str__(self):
        return f"Vegetable: {self._ripe}"

class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    def __str__(self):
        return f"Food: {self._saturation}"

    @property
    def eatable(self):
        return self._saturation > 0


class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Banana(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='yellow', saturation=8):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Pear(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='yellow', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

class Cucumber(Vegetable, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

class Onion(Vegetable, Food):
    def __init__(self, ripe, count=1, max_count=32, color='brown', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

class Inventory():
    def __init__(self, size=10):
        self._list = [None] * size
        self._size = size

    def __str__(self):
        st = ""
        for i in self._list:
            if i is not None:
                st = st + "Ripe: " + str(i._ripe) + ", Saturation: " + str(i._saturation) + ", Count: " + str(i.count) + "; "
        return st

    def put_item(self, idx: int, item: Item):
        if isinstance(item, Item):
            item.update_index(idx)
            self._list[idx] = item
        else:
            raise Exception("Invalid item type.")

    def pick_item(self, item: Item, num):
        if item.index is None:
            raise Exception("There`s no such item in Inventory. Try Inventory.put_item(index, item).")
        elif self._list[item.index] is not None:
            if self._list[item.index].count > num:
                self._list[item.index].update_count(item.count - num)
            elif self._list[item.index].count <= num:
                self._list[item.index] = None
                item.update_index()
    def show(self):
        print(self._list)

class Queue:
    def __init__(self):
        self._items = []

    def show(self):
        print(*self._items)

    def is_empty(self):
        return self._items == []

    def push(self, item):
        self._items.insert(0, item)

    def pop(self):
        tmp = self._items.pop()
        return tmp

    def size(self):
        return len(self._items)

    def copy(self):
        tmp = Queue()
        for i in self._items:
            tmp._items.append(i)
        return tmp


