"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import logging
import time
import numpy as np

from ikats.core.library.exception import IkatsException
from ikats.core.resource.api import IkatsApi
from ikats.core.library.spark import ScManager
from ikats.core.resource.client.temporal_data_mgr import DTYPE

LOGGER = logging.getLogger(__name__)


def dataset_cut(ds_name=None,
                start=None,
                end=None,
                nb_points=None,
                nb_points_by_chunk=50000,
                generate_metadata=False):
    """
    Cutting dataset algorithm.
    Allow to cut a set of TS (dataset).
    This function uses Spark.

    2 methods:
    * Provide a start date and a end date
    * Provide a start date and a number of points to get (much longer - not recommended)

    :param ds_name: name of the dataset to cut
    :param start: start cut date
    :param end: end cut date
    :param nb_points: number of points to cut
    :param nb_points_by_chunk: number of points per chunk
    :param generate_metadata: True to generate metadata on-the-fly (ikats_start_date, ikats_end_date, qual_nb_points)
                              (default : False)

    :type ds_name: str
    :type start: int
    :type end: int or None
    :type nb_points: int or None
    :type generate_metadata: boolean

    :return: list of dict {"tsuid": tsuid, "funcId": func_id}
    :rtype: list of dict

    :raise ValueError: if inputs are not filled properly (see called methods description)
    """

    # Check inputs validity
    if ds_name is None or type(ds_name) is not str:
        raise ValueError('Valid dataset name must be defined (got %s, type: %s)' % (ds_name, type(ds_name)))
    if start is None:
        raise ValueError('Valid start date must be provided (got %s, type: %s)' % (start, type(start)))
    if end is None and nb_points is None:
        raise ValueError('End date or nb points must be provided to cut method')
    if end is not None and nb_points is not None:
        raise ValueError('End date and nb points can not be provided to cut method together')
    if end is not None and start is not None and end == start:
        raise ValueError('start date and end date are identical')

    # List of chunks of data and associated information to parallelize with Spark
    data_to_compute = []

    # Extract tsuid list from input dataset
    tsuid_list = IkatsApi.ds.read(ds_name)['ts_list']

    # Checking metadata availability before starting cutting
    meta_list = IkatsApi.md.read(tsuid_list)

    # Collecting information from metadata
    for tsuid in tsuid_list:
        if tsuid not in meta_list:
            LOGGER.error("Time series %s : no metadata found in base", tsuid)
            raise ValueError("No ikats metadata available for cutting %s" % tsuid)
        if 'ikats_start_date' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("Metadata 'ikats_start_date' for time series %s not found in base", tsuid)
            raise ValueError("No start date available for cutting [%s]" % tsuid)
        if 'ikats_end_date' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("Metadata 'ikats_end_date' for time series %s not found in base", tsuid)
            raise ValueError("No end date available for cutting [%s]" % tsuid)
        if 'qual_ref_period' not in meta_list[tsuid]:
            # Metadata not found
            LOGGER.error("Metadata qual_ref_period' for time series %s not found in base", tsuid)
            raise ValueError("No reference period available for cutting [%s]" % tsuid)

        # Original time series information retrieved from metadata
        sd = int(meta_list[tsuid]['ikats_start_date'])
        ed = int(meta_list[tsuid]['ikats_end_date'])
        ref_period = int(float(meta_list[tsuid]['qual_ref_period']))

        # Get the functional identifier of the original time series
        fid_origin = IkatsApi.ts.fid(tsuid)

        # Generate functional id for resulting time series
        func_id = "%s_cut_%d" % (fid_origin, time.time() * 1e6)

        # Creating new reference in database for new time series
        IkatsApi.ts.create_ref(func_id)

        # Prepare data to compute by defining intervals of final size nb_points_by_chunk
        # Chunk intervals computation :

        data_chunk_size = int(nb_points_by_chunk * ref_period)

        # Computing intervals for chunk definition
        interval_limits = np.hstack(np.arange(sd, ed, data_chunk_size, dtype=np.int64))

        # from intervals we define chunk of data to compute
        # ex : intervals = [ 1, 2, 3] => 2 chunks [1, 2] and [2, 3]
        data_to_compute.extend([(tsuid,
                                 func_id,
                                 i,
                                 interval_limits[i],
                                 interval_limits[i + 1] - 1) for i in range(len(interval_limits) - 1)])
        data_to_compute.append((tsuid,
                                func_id,
                                len(interval_limits) - 1,
                                interval_limits[-1],
                                ed + 1))

    # Review#494 Depending on biggest time series, we could use spark or not and call the former algo or this one below

    LOGGER.info("Running dataset cut using Spark")
    # Create or get a spark Context
    spark_context = ScManager.get()

    try:

        # OUTPUT : [(TSUID_origin, func_id, chunk_index, sd_interval, ed_interval), ...]
        inputs = spark_context.parallelize(data_to_compute, len(data_to_compute))

        # INPUT :  [(TSUID_origin, func_id, chunk_index, sd_interval, ed_interval), ...]
        # OUTPUT : [((TSUID_origin, func_id), chunk_index, original_data_array), ...]
        # PROCESS : read original data in database / filter chunk with no data
        rdd_data = inputs \
            .map(lambda x: ((x[0], x[1]), x[2], IkatsApi.ts.read(tsuid_list=x[0], sd=int(x[3]), ed=int(x[4]))[0])) \
            .filter(lambda x: len(x[2]) > 0)

        # INPUT :  [((TSUID_origin, func_id), chunk_index, original_data_array), ...]
        # OUTPUT : [((TSUID_origin, func_id), chunk_index, (nb_points, data_cut_array)), ...]
        # PROCESS : cut chunks of data, filter empty results
        rdd_cut_chunk_data = rdd_data \
            .map(lambda x: (x[0], x[1], _spark_cut(data=x[2], min_date=start, max_date=end))) \
            .filter(lambda x: len(x[2][1]) > 0) \
            .cache()

        # no end cutting date provided => case of cutting a given number of points
        if end is None:

            # Review#494: The last point of chunkA corresponds to first point of chunkB. I don't see how do you remove this extra point from nb_cumul

            # INPUT : [((TSUID_origin, func_id), chunk_index, (nb_points, data_cut_array)), ...]
            # OUTPUT : [((TSUID_origin, func_id), [(chunk_index1, nb_points1), (chunk_index2, nb_points2),...], ...]
            # PROCESS: Collect nb points associated to chunk indexes
            ts_pts_by_chunk = rdd_cut_chunk_data.map(lambda x: (x[0], (x[1], x[2][0]))) \
                .groupByKey().map(lambda x: (x[0], list(x[1]))) \
                .collect()

            # Compute for each ts from collected data:
            #   - last chunk index containing points to keep
            #   - the number of points to keep in this last chunk
            # cut_info : {(TSUID_origin1, func_id1):(last_chunk_index1, nb_points1),
            #             (TSUID_origin2, func_id2):(last_chunk_index2, nb_points2), ...}
            cut_info = {}
            for ts in ts_pts_by_chunk:
                nb_cumul = 0
                for chunk_index, points in ts[1]:
                    nb_cumul += points
                    # noinspection PyTypeChecker
                    if nb_cumul > nb_points:
                        # noinspection PyTypeChecker
                        cut_info[ts[0]] = (chunk_index, points - (nb_cumul - nb_points))
                        break
                else:
                    LOGGER.warning(
                        "Number of points cut with start cutting date provided exceeds time series %s size"
                        % IkatsApi.ts.fid(ts[0][0]))
                    # case nb_points > nb points of the time series
                    # noinspection PyTypeChecker
                    cut_info[ts[0]] = (chunk_index, points)

            # INPUT : [((TSUID_origin, func_id), chunk_index, (nb_points, data_cut_array)), ...]
            # OUTPUT : [((TSUID_origin, func_id), data_cut_array), ...]
            rdd_cut_data = rdd_cut_chunk_data.filter(lambda x: x[1] <= cut_info[x[0]][0]) \
                .map(lambda x: (x[0], x[2][1][:cut_info[x[0]][1]] if x[1] == cut_info[x[0]][0] else x[2][1]))

        else:
            # INPUT : [((TSUID_origin, func_id), chunk_index, (nb_points, data_cut_array)), ...]
            # OUTPUT : [((TSUID_origin, func_id), data_cut_array), ...]
            rdd_cut_data = rdd_cut_chunk_data.map(lambda x: (x[0], x[2][1]))

        # INPUT :  [((TSUID_origin, func_id), data_cut_array), ...]
        # OUTPUT : [(TSUID_origin, func_id, TSUID, sd, ed), ...]
        # PROCESS : create cut data in database / compute global start and end date
        identifiers = rdd_cut_data \
            .map(lambda x: (x[0][0], x[0][1], _spark_import(fid=x[0][1],
                                                            data=x[1],
                                                            generate_metadata=generate_metadata))) \
            .map(lambda x: ((x[0], x[1], x[2][0]), (x[2][1], x[2][2]))) \
            .reduceByKey(lambda x, y: (min(x[0], y[0]), max(x[1], y[1]))) \
            .map(lambda x: (x[0][0], x[0][1], x[0][2], x[1][0], x[1][1])) \
            .collect()

    except Exception as err:
        msg = "Exception raised while cutting with Spark: %s " % err
        LOGGER.error(msg)
        raise IkatsException(msg)

    finally:
        # Stop spark Context
        ScManager.stop()  # Post-processing : metadata import and return dict building

    # Returns list of dict containing the results of the cut time series : TSUID and functional identifiers
    results = []
    for timeseries in identifiers:
        tsuid_origin = timeseries[0]
        func_id = timeseries[1]
        tsuid = timeseries[2]
        sd = timeseries[3]
        ed = timeseries[4]

        # Review#494: You assume ref_period is always the same for each TS. Moreover, you use the value of the last TS
        # Import metadata in non temporal database
        _save_metadata(tsuid=tsuid, md_name='qual_ref_period', md_value=ref_period, data_type=DTYPE.number,
                       force_update=True)
        _save_metadata(tsuid=tsuid, md_name='ikats_start_date', md_value=sd, data_type=DTYPE.date, force_update=True)
        _save_metadata(tsuid=tsuid, md_name='ikats_end_date', md_value=ed, data_type=DTYPE.date, force_update=True)

        # Retrieve imported number of points from database
        qual_nb_points = IkatsApi.ts.nb_points(tsuid=tsuid)
        IkatsApi.md.create(tsuid=tsuid, name='qual_nb_points', value=qual_nb_points, data_type=DTYPE.number,
                           force_update=True)

        # Inherit from parent
        IkatsApi.ts.inherit(tsuid, tsuid_origin)

        # Fill returned list
        results.append({"tsuid": tsuid, "funcId": func_id})

    return results


def _spark_cut(data, min_date, max_date):
    """
    Performs a temporal cut on data provided :
    keep only data with timestamp greater or equal than min and lesser or equal than max

    NB : last point of data not evaluated

    :param data: data points to cut
    :type data: np.array
    :param min_date: minimum timestamp to keep
    :type min_date: int
    :param max_date: maximum timestamp to keep
    :type max_date: int or None

    :return: number of points after cut, values and timestamps after cut
    :rtype: tuple (int, np.array)
    """

    result = []
    # last point not evaluated
    for timestamp, value in data:
        if timestamp >= min_date:
            # if max not provided, no upper cut performed (case of cutting a number of points)
            if max_date is None or timestamp <= max_date:
                result.append([timestamp, value])

    return len(result), result


def _spark_import(fid, data, generate_metadata):
    """
    Create chunks of data in temporal database

    :param fid: functional identifier
    :param data: data to store in db
    :param generate_metadata: True to generate metadata on the fly while creating data in db

    :type fid: str
    :type data: numpy array
    :type generate_metadata: boolean

    :return: identifier of ts created, start date of chunk, end date of chunk (dates in timestamp)
    :rtype: tuple (tsuid, sd, ed)
    """
    results = IkatsApi.ts.create(fid=fid,
                                 data=data,
                                 generate_metadata=generate_metadata,
                                 sparkified=True)
    start_date = data[0][0]
    end_date = data[-1][0]

    if results['status']:
        return results['tsuid'], start_date, end_date
    else:
        raise IkatsException("TS %s couldn't be created" % fid)


def _save_metadata(tsuid, md_name, md_value, data_type, force_update):
    """
    Saves metadata to Ikats database and log potential errors

    :param tsuid: TSUID to link metadata with
    :param md_name: name of the metadata to save
    :param md_value: value of the metadata
    :param data_type: type of the metadata
    :param force_update: overwrite metadata value if exists (if True)

    :type tsuid: str
    :type md_name: str
    :type md_value: str or int or float
    :type data_type: DTYPE
    :type force_update: bool
    """
    if not IkatsApi.md.create(
            tsuid=tsuid,
            name=md_name,
            value=md_value,
            data_type=data_type,
            force_update=force_update):
        LOGGER.error("Metadata '%s' couldn't be saved for TS %s", md_name, tsuid)
