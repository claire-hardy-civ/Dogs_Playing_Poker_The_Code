## Functions for a general model definition
## Walter Bennette
## June 03, 2020
## walter.bennette.1@us.af.mil
from abc import ABC, abstractmethod 


class General_Model(ABC):
    
    """Class to define a General Model

    This class will enforce that all models have the same basic methods 
    so they can be successfully called using general methods.

    """
    
    @abstractmethod
    def __init__(self, name, options):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, train, val, options):
        
        """Train the model

        """
        
        raise NotImplementedError

    @abstractmethod
    def confidence(self, pool):
        
        """Report model confidence when predicting pool

        """
        
        raise NotImplementedError
    
    
