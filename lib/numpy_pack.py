import numpy as np

def packArray(a):
    """
    pack a numpy array into a bytearray that can be stored as a single 
    field in a spark DataFrame

    :param a: a numpy ndarray 
    :returns: a bytearray
    :rtype:

    """
    if type(a)!=np.ndarray:
        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))
    return bytearray(a.tobytes())

def unpackArray(x,data_type=np.int16):
    """
    unpack a bytearray into a numpy.ndarray, values Smaller than -990 (nominally -999) are mapped to np.nan

    :param x: a bytearray
    :param data_type: The dtype of the array. This is important because if determines how many bytes go into each entry in the array.
    :returns: a numpy array of float16
    :rtype: a numpy ndarray of dtype data_type.

    """
    V=np.frombuffer(x,dtype=data_type)
    V=np.array(V,dtype=np.float16) # Converts to float16, suitable for measurements
    V[V<-990]=np.nan
    return V

# --- UDF Logic based on user's unpackAndScale ---
def unpack_and_scale(measurement_val, values_bytes):
    """
    Unpacks bytearray and scales values.
    Adapted for UDF: takes individual column values as input.
    """
    if values_bytes is None:
        return None

    # Determine data type for unpackArray based on Measurement name
    if '_S' in measurement_val: # As per user's unpackAndScale logic
        v = unpackArray(values_bytes, data_type=np.float16)
    else:
        # unpackArray's default data_type is np.int16, but it internally converts to float16.
        v = unpackArray(values_bytes, data_type=np.int16) 
    
    # Scale if measurement is TMIN, TMAX, or TOBS (as per user's unpackAndScale)
    if measurement_val in ['TMIN', 'TMAX', 'TOBS']:
        v = v / 10.0
    
    # Convert numpy array to a list of Python floats (or None for NaN)
    # posexplode handles nulls within the array gracefully
    return [float(x) if not np.isnan(x) else None for x in v.tolist()]