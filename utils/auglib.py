import random
import typing

import numpy as np


class Augmentations:
    """
    Base class for all augmentations

    p: float, probability of augmentation
    split_func: callable, function to split the text into words
    """

    def __init__(
        self,
        p,
        split_func: typing.Callable = None
    ):
        self.p = p
        if p < 0 or p > 1:
            raise ValueError(f'p must be in [0, 1], got {p}')

        if split_func is None:
            self.split_func = lambda x: x.split()

    def augment(self, inputs):
        raise NotImplementedError

    def __call__(self, text: typing.Union[str, list]) -> typing.Union[str, list]:
        if random.random() < self.p:
            if isinstance(text, str):
                _t = self.split_func(text)
            elif isinstance(text, list):
                _t = text
            else:
                raise ValueError(f'Unknown type of text: {type(text)}')

            _t = self.augment(_t)

            if isinstance(text, str):
                return ' '.join(_t)
            else:
                return _t

        else:
            return text


class Compose(Augmentations):
    """
    Compose several augmentations together
    augmentations: list of augmentations
    """

    def __init__(
        self,
        augmentations: typing.List[Augmentations],
        p=1.0
    ):
        super().__init__(p)
        self.augmentations = augmentations

    def augment(self, text: typing.List[str]) -> typing.List[str]:
        for aug in self.augmentations:
            text = aug(text)
        return text


class OneOf(Augmentations):
    def __init__(
        self,
        augmentations: typing.List[Augmentations],
        p=1.0
    ):
        super().__init__(p)
        self.augmentations = augmentations

    def augment(self, text: typing.List[str]) -> typing.List[str]:
        aug = random.choice(self.augmentations)
        return aug(text)


class InsertWordFromSelf(Augmentations):
    """
    Insert a word from the text into the text
    num_insert: int or float, number of words to insert
        if int, then it is the number of words to insert
        if float, then it is the fraction of words to insert
    repeat_pick: bool, whether to allow picking the same word multiple times
    insert_mode: str, how to insert the word
        'random': insert the word randomly
        'front': insert the word at the front of the text
        'back': insert the word at the back of the text
    """

    def __init__(
        self,
        num_insert: typing.Union[int, float],
        repeat_pick=False,
        insert_mode='random',
        p=0.5,
    ):
        super().__init__(p)
        if isinstance(num_insert, int):
            assert num_insert > 0
        elif isinstance(num_insert, float):
            assert 0 < num_insert < 1

        self.num_insert = num_insert
        self.repeat_pick = repeat_pick

        assert insert_mode in ['random', 'front', 'back']
        self.insert_mode = insert_mode

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_insert, float):
            num_insert = int(len(text) * self.num_insert)
        else:
            num_insert = self.num_insert

        # pick random words from the text
        pick_idx = np.random.choice(len(text), num_insert, replace=self.repeat_pick)
        for i in pick_idx:
            if self.insert_mode == 'front':
                text.insert(0, text[i])
            elif self.insert_mode == 'back':
                text.append(text[i])
            else:  # random
                # insert the word randomly
                text.insert(np.random.randint(len(text)), text[i])

        return text


class InsertWordFromOther(Augmentations):
    def __init__(
        self,
        num_insert: typing.Union[int, float],
        other: list,
        repeat_pick=False,
        insert_mode='random',
        p=0.5
    ):
        super().__init__(p)
        if isinstance(num_insert, int):
            assert num_insert > 0
        elif isinstance(num_insert, float):
            assert 0 < num_insert < 1

        self.num_insert = num_insert
        self.other = other
        self.repeat_pick = repeat_pick

        assert insert_mode in ['random', 'front', 'back']
        self.insert_mode = insert_mode

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_insert, float):
            num_insert = int(len(text) * self.num_insert)
        else:
            num_insert = self.num_insert

        # pick random words from other
        pick_idx = np.random.choice(len(self.other), num_insert, replace=self.repeat_pick)
        for i in pick_idx:
            if self.insert_mode == 'front':
                text.insert(0, self.other[i])
            elif self.insert_mode == 'back':
                text.append(self.other[i])
            else:  # random
                # insert the word randomly
                text.insert(np.random.randint(len(text)), self.other[i])

        return text


class DropWord(Augmentations):
    def __init__(
        self,
        num_drop: typing.Union[int, float],
        p=0.5
    ):
        super().__init__(p)
        if isinstance(num_drop, int):
            assert num_drop > 0
        elif isinstance(num_drop, float):
            assert 0 < num_drop < 1

        self.num_drop = num_drop

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_drop, float):
            num_drop = int(len(text) * self.num_drop)
        else:
            num_drop = self.num_drop

        drop_idx = np.random.choice(len(text), num_drop, replace=False)
        text = [t for i, t in enumerate(text) if i not in drop_idx]

        return text


class RandomSwapWord(Augmentations):
    def __init__(
        self,
        num_swap: typing.Union[int, float],
        p=0.5
    ):
        super().__init__(p)
        if isinstance(num_swap, int):
            assert num_swap > 0
        elif isinstance(num_swap, float):
            assert 0 < num_swap < 1

        self.num_swap = num_swap

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_swap, float):
            num_swap = int(len(text) * self.num_swap)
        else:
            num_swap = self.num_swap

        for _ in range(num_swap):
            idx1, idx2 = np.random.choice(len(text), 2, replace=False)
            text[idx1], text[idx2] = text[idx2], text[idx1]

        return text


class NeighborSwapWord(Augmentations):
    def __init__(
        self,
        num_swap: typing.Union[int, float],
        p=0.5
    ):
        super().__init__(p)
        if isinstance(num_swap, int):
            assert num_swap > 0
        elif isinstance(num_swap, float):
            assert 0 < num_swap < 1

        self.num_swap = num_swap

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_swap, float):
            num_swap = int(len(text) * self.num_swap)
        else:
            num_swap = self.num_swap

        if len(text) < 2:
            return text

        for _ in range(num_swap):
            idx = np.random.choice(len(text) - 1)
            text[idx], text[idx + 1] = text[idx + 1], text[idx]

        return text


class ReplaceWord(Augmentations):
    def __init__(
        self,
        num_replace: typing.Union[int, float],
        replace_func: typing.Callable,
        p=0.5
    ):
        super().__init__(p)
        if isinstance(num_replace, int):
            assert num_replace > 0
        elif isinstance(num_replace, float):
            assert 0 < num_replace < 1

        self.num_replace = num_replace
        self.replace_func = replace_func

    def augment(
        self,
        text: typing.List[str]
    ) -> typing.List[str]:
        if isinstance(self.num_replace, float):
            num_replace = int(len(text) * self.num_replace)
        else:
            num_replace = self.num_replace

        replace_idx = np.random.choice(len(text), num_replace, replace=False)
        for i in replace_idx:
            text[i] = self.replace_func(text[i])

        return text
