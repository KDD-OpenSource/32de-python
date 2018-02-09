from .meta_path_loader import RottenTomatoMetaPathLoader, AbstractMetaPathLoader
from typing import List, Dict

class MetaPathLoaderDispatcher():
    """
    Dispatcher for all currently available meta path data sets.
    """

    available_datasets = [{'name': 'Rotten Tomato',
                           'description': 'Have you ever wondered how similar Arnold Schwarzenegger and all german'
                                          'actors who have appeared in a Sci-Fi movie are? Rotten Tomato is the perfect '
                                          'data set for you!'}]
    dataset_to_loader = {
        'Rotten Tomato': RottenTomatoMetaPathLoader()
    }

    def get_available_datasets(self) -> List[Dict[str, str]]:
        return self.available_datasets

    def get_loader(self, dataset) -> AbstractMetaPathLoader:
        try:
            return self.dataset_to_loader[dataset]
        except KeyError as e:
            print("The data set is not available! ", str(e))