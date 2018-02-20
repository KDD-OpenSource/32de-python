from .meta_path_loader import RottenTomatoMetaPathLoader, AbstractMetaPathLoader, CypherDataSetLoader
from typing import List, Dict
from .config import MOCK_DATASETS_DIR
import os


class MetaPathLoaderDispatcher():
    """
    Dispatcher for all currently available meta path data sets.
    """

    available_datasets = [{'name': 'Rotten Tomato',
                           'description': 'Have you ever wondered how similar Arnold Schwarzenegger and all german'
                                          'actors who have appeared in a Sci-Fi movie are? Rotten Tomato is the perfect '
                                          'data set for you!'},
                          {'name': 'Programming Languages OOvsWeb [Freebase] - length 3',
                           'description': 'How is the world of programming languages connected?' \
                                          'Can ou find the common anchestor of all?' \
                                          'What are the influencial ones?'},
                          {'name': 'Programming Languages OOvsWeb [Freebase] - length 4',
                           'description': 'How is the world of programming languages connected?' \
                                          'Can ou find the common anchestor of all?' \
                                          'What are the influencial ones?'},
                          {'name': 'Programming Languages PHPvsPython [Freebase] - length 5',
                           'description': 'How is the world of programming languages connected?' \
                                          'Can ou find the common anchestor of all?' \
                                          'What are the influencial ones?'}
                          ]
    dataset_to_loader = {
        'Rotten Tomato': RottenTomatoMetaPathLoader(),
        'Programming Languages OOvsWeb [Freebase] - length 3': CypherDataSetLoader(
            os.path.join('..', MOCK_DATASETS_DIR, 'freebase', 'programming_languages', '03-oo_web-1.3.res.csv')),
        'Programming Languages OOvsWeb [Freebase] - length 4': CypherDataSetLoader(
            os.path.join('..', MOCK_DATASETS_DIR, 'freebase', 'programming_languages', '03-oo_web-1.3.res.csv')),
        'Programming Languages PHPvsPython [Freebase] - length 5': CypherDataSetLoader(
            os.path.join('..', MOCK_DATASETS_DIR, 'freebase', 'programming_languages', '01-php_python-1.5.res.csv'))

    }

    def get_available_datasets(self) -> List[Dict[str, str]]:
        return self.available_datasets

    def get_loader(self, dataset) -> AbstractMetaPathLoader:
        try:
            return self.dataset_to_loader[dataset]
        except KeyError as e:
            print("The data set is not available! ", str(e))
