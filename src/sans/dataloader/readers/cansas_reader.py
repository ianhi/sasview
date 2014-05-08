"""
    CanSAS data reader - new recursive cansas_version.
"""
############################################################################
#This software was developed by the University of Tennessee as part of the
#Distributed Data Analysis of Neutron Scattering Experiments (DANSE)
#project funded by the US National Science Foundation. 
#If you use DANSE applications to do scientific research that leads to 
#publication, we ask that you acknowledge the use of the software with the 
#following sentence:
#This work benefited from DANSE software developed under NSF award DMR-0520547. 
#copyright 2008,2009 University of Tennessee
#############################################################################

import logging
import numpy
import os
import sys
import datetime
import inspect
# For saving individual sections of data
from sans.dataloader.data_info import Data1D
from sans.dataloader.data_info import Collimation
from sans.dataloader.data_info import TransmissionSpectrum
from sans.dataloader.data_info import Detector
from sans.dataloader.data_info import Process
from sans.dataloader.data_info import Aperture
# Both imports used. Do not remove either.
from xml.dom.minidom import parseString
import sans.dataloader.readers.xml_reader as xml_reader
from sans.dataloader.readers.xml_reader import XMLreader
from sans.dataloader.readers.cansas_constants import CansasConstants

_ZERO = 1e-16
PREPROCESS = "xmlpreprocess"
ENCODING = "encoding"
HAS_CONVERTER = True
try:
    from sans.data_util.nxsunit import Converter
except:
    HAS_CONVERTER = False

CONSTANTS = CansasConstants()    
CANSAS_FORMAT = CONSTANTS.format
CANSAS_NS = CONSTANTS.names
ALLOW_ALL = True


# minidom used in functions called by outside classes
import xml.dom.minidom
# DO NOT REMOVE
# Called by outside packages:
#    sans.perspectives.invariant.invariant_state
#    sans.perspectives.fitting.pagestate
def get_content(location, node):
    """
    Get the first instance of the content of a xpath location.
    
    :param location: xpath location
    :param node: node to start at
    
    :return: Element, or None
    """
    nodes = node.xpath(location, 
                       namespaces={'ns': CANSAS_NS.get("1.0").get("ns")})
    
    if len(nodes) > 0:
        return nodes[0]
    else:
        return None


# DO NOT REMOVE
# Called by outside packages:
#    sans.perspectives.fitting.pagestate
def write_node(doc, parent, name, value, attr={}):
    """
    :param doc: document DOM
    :param parent: parent node
    :param name: tag of the element
    :param value: value of the child text node
    :param attr: attribute dictionary
    
    :return: True if something was appended, otherwise False
    """
    if value is not None:
        node = doc.createElement(name)
        node.appendChild(doc.createTextNode(str(value)))
        for item in attr:
            node.setAttribute(item, attr[item])
        parent.appendChild(node)
        return True
    return False
                

class Reader(XMLreader):
    """
    Class to load cansas 1D XML files
    
    :Dependencies:
        The CanSAS reader requires PyXML 0.8.4 or later.
    """
    ##CanSAS version - defaults to version 1.0
    cansas_version = "1.0"
    
    logging = []
    errors = []
    
    type_name = "canSAS"
    ## Wildcards
    type = ["XML files (*.xml)|*.xml", "SasView Save Files (*.svs)|*.svs"]
    ## List of allowed extensions
    ext = ['.xml', '.XML', '.svs', '.SVS']
    
    ## Flag to bypass extension check
    allow_all = True
    
    def __init__(self):
        ## List of errors
        self.errors = []
        
    def is_cansas(self, ext="xml"):
        """
        Checks to see if the xml file is a CanSAS file
        """
        if self.validate_xml():
            name = "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
            value = self.xmlroot.get(name)
            if CANSAS_NS.get(self.cansas_version).get("ns") == \
                    value.rsplit(" ")[0]:
                return True
        if ext == "svs":
            return True
        return False
    
    def read(self, xml):
        """
        Validate and read in an xml file in the canSAS format.
        
        :param xml: A canSAS file path in proper XML format
        """
        # X - Q value; Y - Intensity (Abs)
        x_vals = numpy.empty(0)
        y_vals = numpy.empty(0)
        dx_vals = numpy.empty(0)
        dy_vals = numpy.empty(0)
        dxl = numpy.empty(0)
        dxw = numpy.empty(0)
        
        # output - Final list of Data1D objects
        output = []
        # ns - Namespace hierarchy for current xml object
        ns_list = []
        
        # Check that the file exists
        if os.path.isfile(xml):
            basename = os.path.basename(xml)
            _, extension = os.path.splitext(basename)
            # If the file type is not allowed, return nothing
            if extension in self.ext or self.allow_all:
                # Get the file location of 
                base_name = xml_reader.__file__
                base_name = base_name.replace("\\","/")
                base = base_name.split("/sans/")[0]
                
                # Load in xml file and get the cansas version from the header
                self.set_xml_file(xml)
                root = self.xmlroot
                if root is None:
                    root = {}
                self.cansas_version = root.get("version", "1.0")
                
                # Generic values for the cansas file based on the version
                cansas_defaults = CANSAS_NS.get(self.cansas_version, "1.0")
                schema_path = "{0}/sans/dataloader/readers/schema/{1}".format\
                        (base, cansas_defaults.get("schema")).replace("\\", "/")
                
                # Link a schema to the XML file.
                self.set_schema(schema_path)
                
                # Try to load the file, but raise an error if unable to.
                # Check the file matches the XML schema
                try:
                    if self.is_cansas(extension):
                        # Get each SASentry from XML file and add it to a list.
                        entry_list = root.xpath('/ns:SASroot/ns:SASentry',
                                namespaces={'ns': cansas_defaults.get("ns")})
                        ns_list.append("SASentry")
                        
                        # If multiple files, modify the name for each is unique
                        multiple_files = len(entry_list) - 1
                        increment = 0
                        name = basename
                        # Parse each SASentry item
                        for entry in entry_list:
                            # Define a new Data1D object with zeroes for 
                            # x_vals and y_vals
                            data1d = Data1D(x_vals, y_vals, dx_vals, dy_vals)
                            data1d.dxl = dxl
                            data1d.dxw = dxw
                            
                            # If more than one SASentry, increment each in order
                            if multiple_files:
                                name += "_{0}".format(increment)
                                increment += 1
                            
                            # Set the Data1D name and then parse the entry. 
                            # The entry is appended to a list of entry values
                            data1d.filename = name
                            data1d.meta_data["loader"] = "CanSAS 1D"
                            
                            # Get all preprocessing events and encoding
                            self.set_processing_instructions()
                            data1d.meta_data[PREPROCESS] = \
                                    self.processing_instructions
                            
                            # Parse the XML file
                            return_value, extras = \
                                self._parse_entry(entry, ns_list, data1d)
                            del extras[:]
                            
                            # Final cleanup
                            # Remove empty nodes, verify array sizes are correct
                            for error in self.errors:
                                return_value.errors.append(error)
                            del self.errors[:]
                            numpy.trim_zeros(return_value.x)
                            numpy.trim_zeros(return_value.y)
                            numpy.trim_zeros(return_value.dy)
                            size_dx = return_value.dx.size
                            size_dxl = return_value.dxl.size
                            size_dxw = return_value.dxw.size
                            if size_dxl == 0 and size_dxw == 0:
                                return_value.dxl = None
                                return_value.dxw = None
                                numpy.trim_zeros(return_value.dx)
                            elif size_dx == 0:
                                return_value.dx = None
                                size_dx = size_dxl
                                numpy.trim_zeros(return_value.dxl)
                                numpy.trim_zeros(return_value.dxw)
                            output.append(return_value)
                    else:
                        value = self.find_invalid_xml()
                        output.append("Invalid XML at: {0}".format(value))
                except:
                    # If the file does not match the schema, raise this error
                    raise RuntimeError, "%s cannot be read" % xml
                return output
        # Return a list of parsed entries that dataloader can manage
        return None
    
    def _create_unique_key(self, dictionary, name, numb = 0):
        """
        Create a unique key value for any dictionary to prevent overwriting
        Recurses until a unique key value is found.
        
        :param dictionary: A dictionary with any number of entries
        :param name: The index of the item to be added to dictionary
        :param numb: The number to be appended to the name, starts at 0
        """
        if dictionary.get(name) is not None:
            numb += 1
            name = name.split("_")[0]
            name += "_{0}".format(numb)
            name = self._create_unique_key(dictionary, name, numb)
        return name
    
    
    def _unit_conversion(self, new_current_level, attr, data1d, \
                                    tagname, node_value, optional = True):
        """
        A unit converter method used to convert the data included in the file
        to the default units listed in data_info
        
        :param new_current_level: cansas_constants level as returned by 
            iterate_namespace
        :param attr: The attributes of the node
        :param data1d: Where the values will be saved
        :param node_value: The value of the current dom node
        :param optional: Boolean that says if the units are required
        """
        value_unit = ''
        if 'unit' in attr and new_current_level.get('unit') is not None:
            try:
                if isinstance(node_value, float) is False:
                    exec("node_value = float({0})".format(node_value))
                default_unit = None
                unitname = new_current_level.get("unit")
                exec "default_unit = data1d.{0}".format(unitname)
                local_unit = attr['unit']
                if local_unit.lower() != default_unit.lower() and \
                    local_unit is not None and local_unit.lower() != "none" and\
                     default_unit is not None:
                    if HAS_CONVERTER == True:
                        try:
                            ## Check local units - bad units raise KeyError
                            Converter(local_unit)
                            data_conv_q = Converter(attr['unit'])
                            value_unit = default_unit
                            exec "node_value = data_conv_q(node_value, units=data1d.{0})".format(unitname)
                        except KeyError as e:
                            err_msg = "CanSAS reader: could not convert "
                            err_msg += "{0} unit {1}; ".format(tagname, local_unit)
                            intermediate = "err_msg += \"expecting [{1}]  {2}\".format(data1d.{0}, sys.exc_info()[1])".format(unitname, "{0}", "{1}")
                            exec intermediate
                            self.errors.append(err_msg)
                            raise ValueError(err_msg)
                            return
                        except:
                            err_msg = \
                                "CanSAS reader: could not convert the units"
                            self.errors.append(err_msg)
                            return
                    else:
                        value_unit = local_unit
                        err_msg = "CanSAS reader: unrecognized %s unit [%s];"\
                        % (node_value, default_unit)
                        err_msg += " expecting [%s]" % local_unit
                        self.errors.append(err_msg)
                        raise ValueError, err_msg
                        return
                else:
                    value_unit = local_unit
            except:
                err_msg = "CanSAS reader: could not convert "
                err_msg += "Q unit [%s]; " % attr['unit'],
                exec "err_msg += \"expecting [%s]\n  %s\" % (data1d.{0}, sys.exc_info()[1])".format(unitname)
                self.errors.append(err_msg)
                raise ValueError, err_msg
                return
        elif 'unit' in attr:
            value_unit = attr['unit']
        node_value = "float({0})".format(node_value)
        return node_value, value_unit
    
    def _parse_entry(self, dom, names=["SASentry"], data1d=None, extras=[]):
        """
        Parse a SASEntry - new recursive method for parsing the dom of
            the CanSAS data format. This will allow multiple data files
            and extra nodes to be read in simultaneously.
        
        :param dom: dom object with a namespace base of names
        :param names: A list of element names that lead up to the dom object
        :param data1d: The data1d object that will be modified
        :param extras: Any values that should go into meta_data when data1d
            is not a Data1D object
        """
        
        # A portion of every namespace entry
        if data1d == None:
            x_vals = numpy.empty(0)
            y_vals = numpy.empty(0)
            dx_vals = numpy.empty(0)
            dy_vals = numpy.empty(0)
            dxl = numpy.empty(0)
            dxw = numpy.empty(0)
            data1d = Data1D(x_vals, y_vals, dx_vals, dy_vals)
            data1d.dxl = dxl
            data1d.dxw = dxw
                            
        base_ns = "{0}{1}{2}".format("{", \
                            CANSAS_NS.get(self.cansas_version).get("ns"), "}")
        unit = ''
        tagname = ''
        tagname_original = ''
        
        # Go through each child in the parent element
        for node in dom:
            try:
                # Get the element name and set the current names level
                tagname = node.tag.replace(base_ns, "")
                tagname_original = tagname
                if tagname == "fitting_plug_in" or tagname == "pr_inversion" or\
                    tagname == "invariant":
                    continue
                names.append(tagname)
                attr = node.attrib
                children = node.getchildren()
                if len(children) == 0:
                    children = None
                save_data1d = data1d
                
                # Look for special cases
                if tagname == "SASdetector":
                    data1d = Detector()
                elif tagname == "SAScollimation":
                    data1d = Collimation()
                elif tagname == "SAStransmission_spectrum":
                    data1d = TransmissionSpectrum()
                elif tagname == "SASprocess":
                    data1d = Process()
                    for child in node:
                        if child.tag.replace(base_ns, "") == "term":
                            term_attr = {}
                            for attr in child.keys():
                                term_attr[attr] = \
                                    ' '.join(child.get(attr).split())
                            if child.text is not None:
                                term_attr['value'] = \
                                    ' '.join(child.text.split())
                            data1d.term.append(term_attr)
                elif tagname == "aperture":
                    data1d = Aperture()
                if tagname == "Idata" and children is not None:
                    dql = 0
                    dqw = 0
                    for child in children:
                        tag = child.tag.replace(base_ns, "")
                        if tag == "dQl":
                            dql = 1
                        if tag == "dQw":
                            dqw = 1
                    if dqw == 1 and dql == 0:
                        data1d.dxl = numpy.append(data1d.dxl, 0.0)
                    elif dql == 1 and dqw == 0:
                        data1d.dxw = numpy.append(data1d.dxw, 0.0)
                                
                # Get where to store content
                cs_values = CONSTANTS.iterate_namespace(names)
                # If the element is a child element, recurse
                if children is not None:
                    # Returned value is new Data1D object with all previous and 
                    # new values in it.
                    data1d, extras = self._parse_entry(node, 
                                                       names, data1d, extras)
                    
                #Get the information from the node
                node_value = node.text
                if node_value == "":
                    node_value = None
                if node_value is not None:
                    node_value = ' '.join(node_value.split())
                
                # If the value is a float, compile with units.
                if cs_values.ns_datatype == "float":
                    # If an empty value is given, store as zero.
                    if node_value is None or node_value.isspace() \
                                            or node_value.lower() == "nan":
                        node_value = "0.0"
                    node_value, unit = self._unit_conversion(\
                                cs_values.current_level, attr, data1d, \
                                tagname, node_value, cs_values.ns_optional)
                # If the value is a timestamp, convert to a datetime object
                elif cs_values.ns_datatype == "timestamp":
                    if node_value is None or node_value.isspace():
                        pass
                    else:
                        try:
                            node_value = \
                                datetime.datetime.fromtimestamp(node_value)
                        except ValueError:
                            node_value = None
                # If appending to a dictionary (meta_data | run_name)
                # make sure the key is unique
                if cs_values.ns_variable == "{0}.meta_data[\"{2}\"] = \"{1}\"":
                    # If we are within a Process, Detector, Collimation or 
                    # Aperture instance, pull out old data1d
                    tagname = self._create_unique_key(data1d.meta_data, \
                                                      tagname, 0)
                    if isinstance(data1d, Data1D) == False:
                        store_me = cs_values.ns_variable.format("data1d", \
                                                            node_value, tagname)
                        extras.append(store_me)
                        cs_values.ns_variable = None
                if cs_values.ns_variable == "{0}.run_name[\"{2}\"] = \"{1}\"":
                    tagname = self._create_unique_key(data1d.run_name, \
                                                      tagname, 0)
                
                # Check for Data1D object and any extra commands to save
                if isinstance(data1d, Data1D):
                    for item in extras:
                        exec item
                # Don't bother saving empty information unless it is a float
                if cs_values.ns_variable is not None and \
                            node_value is not None and \
                            node_value.isspace() == False:
                    # Format a string and then execute it.
                    store_me = cs_values.ns_variable.format("data1d", \
                                                            node_value, tagname)
                    exec store_me
                # Get attributes and process them
                if attr is not None:
                    for key in node.keys():
                        try:
                            cansas_attrib = \
                            cs_values.current_level.get("attributes").get(key)
                            attrib_variable = cansas_attrib.get("variable")
                            if key == 'unit' and unit != '':
                                attrib_value = unit
                            else:
                                attrib_value = node.attrib[key]
                            store_attr = attrib_variable.format("data1d", \
                                                            attrib_value, key)
                            exec store_attr
                        except AttributeError as e:
                            pass
            
            except TypeError as e:
                pass
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_type, fname, exc_tb.tb_lineno, tagname, exc_obj)
            finally:
                # Save special cases in original data1d object
                # then restore the data1d
                if tagname_original == "SASdetector":
                    save_data1d.detector.append(data1d)
                elif tagname_original == "SAScollimation":
                    save_data1d.collimation.append(data1d)
                elif tagname == "SAStransmission_spectrum":
                    save_data1d.trans_spectrum.append(data1d)
                elif tagname_original == "SASprocess":
                    save_data1d.process.append(data1d)
                elif tagname_original == "aperture":
                    save_data1d.aperture.append(data1d)
                else:
                    save_data1d = data1d
                if tagname_original == "fitting_plug_in" or \
                    tagname_original == "invariant" or \
                    tagname_original == "pr_inversion":
                    pass
                else:
                    data1d = save_data1d
                    # Remove tagname from names to restore original base
                    names.remove(tagname_original)
        return data1d, extras
        
    
    def _to_xml_doc(self, datainfo):
        """
        Create an XML document to contain the content of a Data1D
        
        :param datainfo: Data1D object
        """
        
        if not issubclass(datainfo.__class__, Data1D):
            raise RuntimeError, "The cansas writer expects a Data1D instance"
        
        # Get PIs and create root element
        pis = self.return_processing_instructions()
        if len(pis) > 0:
            pi_tree = self.create_tree(pis[0])
            i = 1
            for i in range(1,len(pis) - 1):
                pi_tree = self.append(pis[i], pi_tree)
            pi_string = self.to_string(pi_tree)
        else:
            pi_string = ""
        
        # Define namespaces and create SASroot object
        xsi = "http://www.w3.org/2001/XMLSchema-instance"
        version = self.cansas_version
        ns = CANSAS_NS.get(version).get("ns")
        if version == "1.1":
            url = "http://www.cansas.org/formats/1.1/"
        else:
            url = "http://svn.smallangles.net/svn/canSAS/1dwg/trunk/"
        schema_location = "{0} {1}cansas1d.xsd".format(ns, url)
        attrib = {"{" + xsi + "}schemaLocation" : schema_location,
                  "version" : version}
        nsmap = {'xsi' : xsi, None: ns}
        
        main_node = self.create_element("{" + ns + "}SASroot", \
                                               attrib = attrib, \
                                               nsmap = nsmap)
        
        # Create ElementTree, append SASroot and apply processing instructions
        base_string = pi_string + self.to_string(main_node)
        base_element = self.create_element_from_string(base_string)
        doc = self.create_tree(base_element)
        
        # Create SASentry Element
        entry_node = self.create_element("SASentry")
        root = doc.getroot()
        root.append(entry_node)
        
        # Add Title to SASentry
        self.write_node(entry_node, "Title", datainfo.title)
        
        # Add Run to SASentry
        if datainfo.run == None or datainfo.run == []:
            RUN_NAME_DEFAULT = "None"
            datainfo.run.append(RUN_NAME_DEFAULT)
            datainfo.run_name[RUN_NAME_DEFAULT] = RUN_NAME_DEFAULT
        for item in datainfo.run:
            runname = {}
            if item in datainfo.run_name and \
            len(str(datainfo.run_name[item])) > 1:
                runname = {'name': datainfo.run_name[item]}
            self.write_node(entry_node, "Run", item, runname)
        
        # Data info
        node = self.create_element("SASdata")
        self.append(node, entry_node)
        
        for i in range(len(datainfo.x)):
            pt = self.create_element("Idata")
            node.append(pt)
            self.write_node(pt, "Q", datainfo.x[i], {'unit': datainfo.x_unit})
            if len(datainfo.y) >= i:
                self.write_node(pt, "I", datainfo.y[i],
                            {'unit': datainfo.y_unit})
            if datainfo.dy != None and len(datainfo.dy) > i:
                self.write_node(pt, "Idev", datainfo.dy[i],
                            {'unit': datainfo.y_unit})
            if datainfo.dx != None and len(datainfo.dx) > i:
                self.write_node(pt, "Qdev", datainfo.dx[i],
                            {'unit': datainfo.x_unit})
            if datainfo.dxw != None and len(datainfo.dxw) > i:
                self.write_node(pt, "dQw", datainfo.dxw[i],
                            {'unit': datainfo.x_unit})
            if datainfo.dxl != None and len(datainfo.dxl) > i:
                self.write_node(pt, "dQl", datainfo.dxl[i],
                            {'unit': datainfo.x_unit})
        
        # Transmission Spectrum Info
        for i in range(len(datainfo.trans_spectrum)):
            spectrum = datainfo.trans_spectrum[i]
            node = self.create_element("SAStransmission_spectrum",
                                              {"name" : spectrum.name})
            self.append(node, entry_node)
            if isinstance(spectrum.timestamp, datetime.datetime):
                node.setAttribute("timestamp", spectrum.timestamp)
            for i in range(len(spectrum.wavelength)):
                pt = self.create_element("Tdata")
                node.append(pt)
                self.write_node(pt, "Lambda", spectrum.wavelength[i], 
                           {'unit': spectrum.wavelength_unit})
                self.write_node(pt, "T", spectrum.transmission[i], 
                           {'unit': spectrum.transmission_unit})
                if spectrum.transmission_deviation != None \
                and len(spectrum.transmission_deviation) >= i:
                    self.write_node(pt, "Tdev", \
                               spectrum.transmission_deviation[i], \
                               {'unit': spectrum.transmission_deviation_unit})

        # Sample info
        sample = self.create_element("SASsample")
        if datainfo.sample.name is not None:
            self.write_attribute(sample, 
                                        "name", 
                                        str(datainfo.sample.name))
        self.append(sample, entry_node)
        self.write_node(sample, "ID", str(datainfo.sample.ID))
        self.write_node(sample, "thickness", datainfo.sample.thickness,
                   {"unit": datainfo.sample.thickness_unit})
        self.write_node(sample, "transmission", datainfo.sample.transmission)
        self.write_node(sample, "temperature", datainfo.sample.temperature,
                   {"unit": datainfo.sample.temperature_unit})
        
        pos = self.create_element("position")
        written = self.write_node(pos, 
                                  "x", 
                                  datainfo.sample.position.x,
                                  {"unit": datainfo.sample.position_unit})
        written = written | self.write_node(pos, 
                                            "y", 
                                            datainfo.sample.position.y,
                                       {"unit": datainfo.sample.position_unit})
        written = written | self.write_node(pos, 
                                            "z",
                                            datainfo.sample.position.z,
                                       {"unit": datainfo.sample.position_unit})
        if written == True:
            self.append(pos, sample)
        
        ori = self.create_element("orientation")
        written = self.write_node(ori, "roll",
                                  datainfo.sample.orientation.x,
                                  {"unit": datainfo.sample.orientation_unit})
        written = written | self.write_node(ori, "pitch",
                                       datainfo.sample.orientation.y,
                                    {"unit": datainfo.sample.orientation_unit})
        written = written | self.write_node(ori, "yaw",
                                       datainfo.sample.orientation.z,
                                    {"unit": datainfo.sample.orientation_unit})
        if written == True:
            self.append(ori, sample)
        
        for item in datainfo.sample.details:
            self.write_node(sample, "details", item)
        
        # Instrument info
        instr = self.create_element("SASinstrument")
        self.append(instr, entry_node)
        
        self.write_node(instr, "name", datainfo.instrument)
        
        #   Source
        source = self.create_element("SASsource")
        if datainfo.source.name is not None:
            self.write_attribute(source,
                                        "name",
                                        str(datainfo.source.name))
        self.append(source, instr)
        if datainfo.source.radiation == None or datainfo.source.radiation == '':
            datainfo.source.radiation = "neutron"
        self.write_node(source, "radiation", datainfo.source.radiation)
        
        size = self.create_element("beam_size")
        if datainfo.source.beam_size_name is not None:
            self.write_attribute(size,
                                        "name",
                                        str(datainfo.source.beam_size_name))
        written = self.write_node(size, "x", datainfo.source.beam_size.x,
                             {"unit": datainfo.source.beam_size_unit})
        written = written | self.write_node(size, "y",
                                       datainfo.source.beam_size.y,
                                       {"unit": datainfo.source.beam_size_unit})
        written = written | self.write_node(size, "z",
                                       datainfo.source.beam_size.z,
                                       {"unit": datainfo.source.beam_size_unit})
        if written == True:
            self.append(size, source)
            
        self.write_node(source, "beam_shape", datainfo.source.beam_shape)
        self.write_node(source, "wavelength",
                   datainfo.source.wavelength,
                   {"unit": datainfo.source.wavelength_unit})
        self.write_node(source, "wavelength_min",
                   datainfo.source.wavelength_min,
                   {"unit": datainfo.source.wavelength_min_unit})
        self.write_node(source, "wavelength_max",
                   datainfo.source.wavelength_max,
                   {"unit": datainfo.source.wavelength_max_unit})
        self.write_node(source, "wavelength_spread",
                   datainfo.source.wavelength_spread,
                   {"unit": datainfo.source.wavelength_spread_unit})
        
        #   Collimation
        if datainfo.collimation == [] or datainfo.collimation == None:
            coll = Collimation()
            datainfo.collimation.append(coll)
        for item in datainfo.collimation:
            coll = self.create_element("SAScollimation")
            if item.name is not None:
                self.write_attribute(coll, "name", str(item.name))
            self.append(coll, instr)
            
            self.write_node(coll, "length", item.length,
                       {"unit": item.length_unit})
            
            for apert in item.aperture:
                ap = self.create_element("aperture")
                if apert.name is not None:
                    self.write_attribute(ap, "name", str(apert.name))
                if apert.type is not None:
                    self.write_attribute(ap, "type", str(apert.type))
                self.append(ap, coll)
                
                size = self.create_element("size")
                if apert.size_name is not None:
                    self.write_attribute(size, 
                                                "name", 
                                                str(apert.size_name))
                written = self.write_node(size, "x", apert.size.x,
                                     {"unit": apert.size_unit})
                written = written | self.write_node(size, "y", apert.size.y,
                                               {"unit": apert.size_unit})
                written = written | self.write_node(size, "z", apert.size.z,
                                               {"unit": apert.size_unit})
                if written == True:
                    self.append(size, ap)
                
                self.write_node(ap, "distance", apert.distance,
                           {"unit": apert.distance_unit})

        #   Detectors
        if datainfo.detector == None or datainfo.detector == []:
            det = Detector()
            det.name = ""
            datainfo.detector.append(det)
                
        for item in datainfo.detector:
            det = self.create_element("SASdetector")
            written = self.write_node(det, "name", item.name)
            written = written | self.write_node(det, "SDD", item.distance,
                                           {"unit": item.distance_unit})
            if written == True:
                self.append(det, instr)
            
            off = self.create_element("offset")
            written = self.write_node(off, "x", item.offset.x,
                                 {"unit": item.offset_unit})
            written = written | self.write_node(off, "y", item.offset.y,
                                           {"unit": item.offset_unit})
            written = written | self.write_node(off, "z", item.offset.z,
                                           {"unit": item.offset_unit})
            if written == True:
                self.append(off, det)
                
            ori = self.create_element("orientation")
            written = self.write_node(ori, "roll", item.orientation.x,
                                 {"unit": item.orientation_unit})
            written = written | self.write_node(ori, "pitch",
                                           item.orientation.y,
                                           {"unit": item.orientation_unit})
            written = written | self.write_node(ori, "yaw",
                                           item.orientation.z,
                                           {"unit": item.orientation_unit})
            if written == True:
                self.append(ori, det)
            
            center = self.create_element("beam_center")
            written = self.write_node(center, "x", item.beam_center.x,
                                 {"unit": item.beam_center_unit})
            written = written | self.write_node(center, "y",
                                           item.beam_center.y,
                                           {"unit": item.beam_center_unit})
            written = written | self.write_node(center, "z",
                                           item.beam_center.z,
                                           {"unit": item.beam_center_unit})
            if written == True:
                self.append(center, det)
                
            pix = self.create_element("pixel_size")
            written = self.write_node(pix, "x", item.pixel_size.x,
                                 {"unit": item.pixel_size_unit})
            written = written | self.write_node(pix, "y", item.pixel_size.y,
                                           {"unit": item.pixel_size_unit})
            written = written | self.write_node(pix, "z", item.pixel_size.z,
                                           {"unit": item.pixel_size_unit})
            written = written | self.write_node(det, "slit_length",
                                           item.slit_length,
                                           {"unit": item.slit_length_unit})
            if written == True:
                self.append(pix, det)
            
        # Processes info
        for item in datainfo.process:
            node = self.create_element("SASprocess")
            self.append(node, entry_node)

            self.write_node(node, "name", item.name)
            self.write_node(node, "date", item.date)
            self.write_node(node, "description", item.description)
            for term in item.term:
                value = term['value']
                del term['value']
                self.write_node(node, "term", value, term)
            for note in item.notes:
                self.write_node(node, "SASprocessnote", note)
            if len(item.notes) == 0:
                self.write_node(node, "SASprocessnote", "")
                
        # Note info
        if len(datainfo.notes) == 0:
            node = self.create_element("SASnote")
            self.append(node, entry_node)
        else:
            for item in datainfo.notes:
                node = self.create_element("SASnote")
                self.write_text(node, item)
                self.append(node, entry_node)
                
        
        # Return the document, and the SASentry node associated with
        # the data we just wrote
        
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        mod_name = mod.__name__
        if mod_name != "sans.dataloader.readers.cansas_reader":
            string = self.to_string(doc, pp=True)
            doc = parseString(string)
            node_name = entry_node.tag
            node_list = doc.getElementsByTagName(node_name)
            entry_node = node_list.item(0)
        
        return doc, entry_node
    
    
    def write_node(self, parent, name, value, attr=None):
        """
        :param doc: document DOM
        :param parent: parent node
        :param name: tag of the element
        :param value: value of the child text node
        :param attr: attribute dictionary
        
        :return: True if something was appended, otherwise False
        """
        if value is not None:
            parent = self.ebuilder(parent, name, value, attr)
            return True
        return False
    
            
    def write(self, filename, datainfo):
        """
        Write the content of a Data1D as a CanSAS XML file
        
        :param filename: name of the file to write
        :param datainfo: Data1D object
        """
        # Create XML document
        doc, _ = self._to_xml_doc(datainfo)
        # Write the file
        fd = open(filename, 'w')
        if self.encoding == None:
            self.encoding = "UTF-8"
        doc.write(fd, encoding=self.encoding,
                  pretty_print=True, xml_declaration=True)
        fd.close()
    
    
    # DO NOT REMOVE - used in saving and loading panel states.
    def _store_float(self, location, node, variable, storage, optional=True):
        """
        Get the content of a xpath location and store
        the result. Check that the units are compatible
        with the destination. The value is expected to
        be a float.
        
        The xpath location might or might not exist.
        If it does not exist, nothing is done
       
        :param location: xpath location to fetch
        :param node: node to read the data from
        :param variable: name of the data member to store it in [string]
        :param storage: data object that has the 'variable' data member
        :param optional: if True, no exception will be raised
            if unit conversion can't be done

        :raise ValueError: raised when the units are not recognized
        """
        entry = get_content(location, node)
        try:
            value = float(entry.text)
        except:
            value = None
            
        if value is not None:
            # If the entry has units, check to see that they are
            # compatible with what we currently have in the data object
            units = entry.get('unit')
            if units is not None:
                toks = variable.split('.')
                local_unit = None
                exec "local_unit = storage.%s_unit" % toks[0]
                if local_unit != None and units.lower() != local_unit.lower():
                    if HAS_CONVERTER == True:
                        try:
                            conv = Converter(units)
                            exec "storage.%s = %g" % (variable,
                                           conv(value, units=local_unit))
                        except:
                            err_mess = "CanSAS reader: could not convert"
                            err_mess += " %s unit [%s]; expecting [%s]\n  %s" \
                                % (variable, units, local_unit, sys.exc_value)
                            self.errors.append(err_mess)
                            if optional:
                                logging.info(err_mess)
                            else:
                                raise ValueError, err_mess
                    else:
                        err_mess = "CanSAS reader: unrecognized %s unit [%s];"\
                        % (variable, units)
                        err_mess += " expecting [%s]" % local_unit
                        self.errors.append(err_mess)
                        if optional:
                            logging.info(err_mess)
                        else:
                            raise ValueError, err_mess
                else:
                    exec "storage.%s = value" % variable
            else:
                exec "storage.%s = value" % variable
                
    
    # DO NOT REMOVE - used in saving and loading panel states.
    def _store_content(self, location, node, variable, storage):
        """
        Get the content of a xpath location and store
        the result. The value is treated as a string.
        
        The xpath location might or might not exist.
        If it does not exist, nothing is done
        
        :param location: xpath location to fetch
        :param node: node to read the data from
        :param variable: name of the data member to store it in [string]
        :param storage: data object that has the 'variable' data member
        
        :return: return a list of errors
        """
        entry = get_content(location, node)
        if entry is not None and entry.text is not None:
            exec "storage.%s = entry.text.strip()" % variable