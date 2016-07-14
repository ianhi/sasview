""" 
Provide base functionality for all model components
"""

# imports   
import copy
import numpy
#TO DO: that about a way to make the parameter
#is self return if it is fittable or not  

class BaseComponent(object):
    """ 
    Basic model component patched with
    Sasview wrapper for opencl/ctypes model.
    """
    # Model parameters for the specific model are set in the class constructor
    # via the _generate_model_attributes function, which subclasses
    # SasviewModel.  They are included here for typing and documentation
    # purposes.
    _model = None       # type: KernelModel
    _model_info = None  # type: ModelInfo
    #: load/save name for the model
    id = None           # type: str
    #: display name for the model
    name = None         # type: str
    #: short model description
    description = None  # type: str
    #: default model category
    category = None     # type: str

    #: names of the orientation parameters in the order they appear
    orientation_params = None # type: Sequence[str]
    #: names of the magnetic parameters in the order they appear
    magnetic_params = None    # type: Sequence[str]
    #: names of the fittable parameters
    fixed = None              # type: Sequence[str]
    # TODO: the attribute fixed is ill-named

    # Axis labels
    input_name = "Q"
    input_unit = "A^{-1}"
    output_name = "Intensity"
    output_unit = "cm^{-1}"

    #: default cutoff for polydispersity
    cutoff = 1e-5

    # Note: Use non-mutable values for class attributes to avoid errors
    #: parameters that are not fitted
    non_fittable = ()        # type: Sequence[str]

    #: True if model should appear as a structure factor
    is_structure_factor = False
    #: True if model should appear as a form factor
    is_form_factor = False
    #: True if model has multiplicity
    is_multiplicity_model = False
    #: Mulitplicity information
    multiplicity_info = None # type: MultiplicityInfoType

    # Per-instance variables
    #: parameter {name: value} mapping
    params = None      # type: Dict[str, float]
    #: values for dispersion width, npts, nsigmas and type
    dispersion = None  # type: Dict[str, Any]
    #: units and limits for each parameter
    details = None     # type: Mapping[str, Tuple(str, float, float)]
    #: multiplicity used, or None if no multiplicity controls
    multiplicity = None     # type: Optional[int]


    def __init__(self, multiplicity):
        """ Initialization"""

        # type: () -> None
        #print("initializing", self.name)
        #raise Exception("first initialization")
        self._model = None

        ## _persistency_dict is used by sas.perspectives.fitting.basepage
        ## to store dispersity reference.
        self._persistency_dict = {}

        self.multiplicity = multiplicity

        #self.params = collections.OrderedDict()
        #self.dispersion = collections.OrderedDict()
        self.params = {}
        self.dispersion = {}
        self.details = {}

        for p in self._model_info['parameters']:
            self.params[p.name] = p.default
            self.details[p.name] = [p.units] + p.limits

        for name in self._model_info['partype']['pd-2d']:
             self.dispersion[name] = {
                 'width': 0,
                 'npts': 35,
                 'nsigmas': 3,
                 'type': 'gaussian',
             }


    def __get_state__(self):
        state = self.__dict__.copy()
        state.pop('_model')
        # May need to reload model info on set state since it has pointers
        # to python implementations of Iq, etc.
        #state.pop('_model_info')
        return state

    def __set_state__(self, state):
        self.__dict__ = state
        self._model = None

    def __str__(self):
        """ 
        :return: string representation
        """
        return self.name
   
    def is_fittable(self, par_name):
        """
        Check if a given parameter is fittable or not
        
        :param par_name: the parameter name to check
        """
        return par_name in self.fixed
        #For the future
        #return self.params[str(par_name)].is_fittable()
   
    def run(self, x): 
        """
        run 1d
        """
        return NotImplemented
    
    def runXY(self, x): 
        """
        run 2d
        """
        return NotImplemented  
    
    def calculate_ER(self): 
        """
        Calculate effective radius
        """
        return NotImplemented  
    
    def calculate_VR(self): 
        """
        Calculate volume fraction ratio
        """
        return NotImplemented 
    
    def evalDistribution(self, qdist):
        """
        Evaluate a distribution of q-values.
        
        * For 1D, a numpy array is expected as input: ::
        
            evalDistribution(q)
            
          where q is a numpy array.
        
        
        * For 2D, a list of numpy arrays are expected: [qx_prime,qy_prime],
          where 1D arrays, ::
        
              qx_prime = [ qx[0], qx[1], qx[2], ....]

          and ::

              qy_prime = [ qy[0], qy[1], qy[2], ....] 
        
        Then get ::

            q = numpy.sqrt(qx_prime^2+qy_prime^2)
        
        that is a qr in 1D array; ::

            q = [q[0], q[1], q[2], ....] 
        
        ..note::
          Due to 2D speed issue, no anisotropic scattering 
          is supported for python models, thus C-models should have
          their own evalDistribution methods.
        
        The method is then called the following way: ::
        
            evalDistribution(q)

        where q is a numpy array.
        
        :param qdist: ndarray of scalar q-values or list [qx,qy] where qx,qy are 1D ndarrays
        """
        if qdist.__class__.__name__ == 'list':
            # Check whether we have a list of ndarrays [qx,qy]
            if len(qdist)!=2 or \
                qdist[0].__class__.__name__ != 'ndarray' or \
                qdist[1].__class__.__name__ != 'ndarray':
                msg = "evalDistribution expects a list of 2 ndarrays"
                raise RuntimeError, msg
                
            # Extract qx and qy for code clarity
            qx = qdist[0]
            qy = qdist[1]
            
            # calculate q_r component for 2D isotropic
            q = numpy.sqrt(qx**2+qy**2)
            # vectorize the model function runXY
            v_model = numpy.vectorize(self.runXY, otypes=[float])
            # calculate the scattering
            iq_array = v_model(q)

            return iq_array
                
        elif qdist.__class__.__name__ == 'ndarray':
            # We have a simple 1D distribution of q-values
            v_model = numpy.vectorize(self.runXY, otypes=[float])
            iq_array = v_model(qdist)
            return iq_array
            
        else:
            mesg = "evalDistribution is expecting an ndarray of scalar q-values"
            mesg += " or a list [qx,qy] where qx,qy are 2D ndarrays."
            raise RuntimeError, mesg
        
    
    
    def clone(self):
        """ Returns a new object identical to the current object """
        obj = copy.deepcopy(self)
        return self._clone(obj)
    
    def _clone(self, obj):
        """
        Internal utility function to copy the internal
        data members to a fresh copy.
        """
        obj.params     = copy.deepcopy(self.params)
        obj.details    = copy.deepcopy(self.details)
        obj.dispersion = copy.deepcopy(self.dispersion)
        obj._persistency_dict = copy.deepcopy( self._persistency_dict)
        return obj
    
    def set_dispersion(self, parameter, dispersion):
        """
        model dispersions
        """ 
        ##Not Implemented
        return None
        
    def getProfile(self):
        """
        Get SLD profile 
        
        : return: (z, beta) where z is a list of depth of the transition points
                beta is a list of the corresponding SLD values 
        """
        #Not Implemented
        return None, None
            
    def setParam(self, name, value):
        """ 
        Set the value of a model parameter
    
        :param name: name of the parameter
        :param value: value of the parameter
        
        """
        # Look for dispersion parameters
        toks = name.split('.')
        if len(toks) == 2:
            for item in self.dispersion.keys():
                if item == toks[0]:
                    for par in self.dispersion[item]:
                        if par == toks[1]:
                            self.dispersion[item][par] = value
                            return
        else:
            # Look for standard parameter
            for item in self.params.keys():
                if item == name:
                    self.params[item] = value
                    return

        raise ValueError("Model does not contain parameter %s" % name)
        
    def getParam(self, name):
        """ 
        Set the value of a model parameter

        :param name: name of the parameter
        
        """
        # Look for dispersion parameters
        toks = name.split('.')
        if len(toks) == 2:
            for item in self.dispersion.keys():
                if item == toks[0]:
                    for par in self.dispersion[item]:
                        if par == toks[1]:
                            return self.dispersion[item][par]
        else:
            # Look for standard parameter
            for item in self.params.keys():
                if item == name:
                    return self.params[item]

        raise ValueError("Model does not contain parameter %s" % name)

    def getParamList(self):
        """ 
        Return a list of all available parameters for the model
        """ 
        param_list = self.params.keys()
        # WARNING: Extending the list with the dispersion parameters
        param_list.extend(self.getDispParamList())
        return param_list
    
    def getDispParamList(self):
        """ 
        Return a list of all available parameters for the model
        """ 
        list = []
        
        for item in self.dispersion.keys():
            for p in self.dispersion[item].keys():
                if p not in ['type']:
                    list.append('%s.%s' % (item.lower(), p.lower()))
                    
        return list
    
    # Old-style methods that are no longer used
    def setParamWithToken(self, name, value, token, member): 
        """
        set Param With Token
        """
        return NotImplemented
    def getParamWithToken(self, name, token, member): 
        """
        get Param With Token
        """
        return NotImplemented
    
    def getParamListWithToken(self, token, member): 
        """
        get Param List With Token
        """
        return NotImplemented
