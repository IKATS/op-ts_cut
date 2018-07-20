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
import unittest

import numpy as np

from ikats.algo.ts_cut.ds_cut import dataset_cut
from ikats.core.resource.api import IkatsApi


def log_to_stdout(logger_to_use):
    """
    Allow to print some loggers to stdout
    :param logger_to_use: the LOGGER object to redirect to stdout
    """

    logger_to_use.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger_to_use.addHandler(stream_handler)


def gen_ts(data, ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param data: data points to use for TS creation
    :type data: ndarray

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID and funcId
    :rtype: tuple
    """

    # Build TS identifier
    fid = "UNIT_TEST_Cut_DS_%s" % ts_id

    # Remove former potential time series having this name
    try:
        tsuid = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
    except ValueError:
        # No TS to delete
        pass

    # Create the time series
    result = IkatsApi.ts.create(fid=fid, data=np.array(data))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(data), force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return result['tsuid'], fid


def _check_results(self, ts_list, result, expected_data):
    """
    Check the results of the test and compare it to the expected data

    :param ts_list: list of duet tsuid/funcId to match input to output
    :param result: raw result of the operator
    :param expected_data: expected data to be used as comparison reference

    :type ts_list: list of dict
    :type result: dict
    :type expected_data: dict

    :raise Exception: if any check fails
    """

    # Check number of results is the same
    self.assertEqual(len(ts_list), len(result))

    # Check data content
    for index, ts_item in enumerate(ts_list):
        original_tsuid = ts_item["tsuid"]
        obtained_tsuid = result[original_tsuid]["tsuid"]
        obtained_data = IkatsApi.ts.read(tsuid_list=[obtained_tsuid])[0]

        # Compare values
        try:
            self.assertTrue(np.allclose(
                np.array(expected_data[original_tsuid], dtype=np.float64),
                np.array(obtained_data, dtype=np.float64),
                atol=1e-2))
        except Exception:
            print("ts_item:%s" % ts_item)
            print("Expected (%d points)" % len(expected_data[original_tsuid]))
            print(expected_data[original_tsuid])
            print("Obtained (%d points)" % len(obtained_data))
            print(obtained_data)
            raise


def _init_nominal(ts_id=1):
    data_to_cut = np.array([
        [1e12 + 1000, 3.0],
        [1e12 + 2000, 15.0],
        [1e12 + 3000, 8.0],
        [1e12 + 6000, 25.89],
        [1e12 + 8000, 3.0],
        [1e12 + 9000, 21.2],
        [1e12 + 40000, 18],
        [1e12 + 43000, 15.0],
        [1e12 + 43500, 12.0],
        [1e12 + 44000, 7.5],
        [1e12 + 52000, 35.0]])

    return gen_ts(data_to_cut, ts_id)


class TestDsCut(unittest.TestCase):
    """
    Test of temporal cut
    """

    # Review#494 Missing test case : 1 of all TS has no points in range
    # Review#494 Missing test case : Dataset unknown
    # Review#494 Missing test case : Dataset has no TS
    # Review#494 Missing test case : start date not aligned with point date
    # Review#494 Missing test case : end date not aligned with point date
    # Review#494 Missing test case : number of points too big compared to 1 TS points count
    # Review#494 Missing test case : end date < start date
    # Review#494 Missing test case : number of points <0

    def array_equality(self, expected_data, tsuid_result):
        self.assertTrue(np.allclose(
            np.array(expected_data, dtype=np.float64),
            np.array(IkatsApi.ts.read(tsuid_list=tsuid_result)[0], dtype=np.float64),
            atol=1e-3))

    def test_nominal_nb_points(self):
        """
        Compute the cut on a single time series
        from start date and with a number of points provided
        Check the time series is cut as expected

        case : NOMINAL
        """

        tsuid, fid = _init_nominal()
        start_cut = int(1e12 + 3000)
        end_cut = None
        nb_points_cut = 7
        ds_name = "DS_Test_Cut_Dataset"
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 3000, 8.0],
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0]
        ])
        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[tsuid])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)
            result_tsuids = [x['tsuid'] for x in result]
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        finally:
            # clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_nominal_end_date(self):
        """
        Compute the cut on a single time series
        between start and end date
        Check the time series is cut as expected

        case : NOMINAL
        """

        tsuid, fid = _init_nominal()
        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 44000)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset"
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 3000, 8.0],
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0],
            [1e12 + 44000, 7.5]
        ])

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[tsuid])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)
            result_tsuids = [x['tsuid'] for x in result]
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        finally:
            # clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_nominal_with_chunks(self):
        """
        Compute the cut on a single time series
        between start and end date with several chunks of data
        Check the time series is cut as expected

        case : NOMINAL
        """

        tsuid, fid = _init_nominal()
        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 44000)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset"
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 3000, 8.0],
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0],
            [1e12 + 44000, 7.5]
        ])

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[tsuid])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut,
                                 nb_points_by_chunk=2)
            result_tsuids = [x['tsuid'] for x in result]
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        finally:
            # Clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_dataset_multi_ts(self):
        """
        Compute the cut on a multi time series dataset
        between start and end date with several chunks of data
        Check the time series are cut as expected

        case : NOMINAL
        """

        tsuid1, fid1 = _init_nominal(1)
        tsuid2, fid2 = _init_nominal(2)
        tsuid3, fid3 = _init_nominal(3)
        tsuid4, fid4 = _init_nominal(4)
        tsuid5, fid5 = _init_nominal(5)

        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 44000)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset"
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 3000, 8.0],
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0],
            [1e12 + 44000, 7.5]
        ])

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[tsuid1, tsuid2, tsuid3, tsuid4, tsuid5])
        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut,
                                 nb_points_by_chunk=7)
            result_tsuids = [x['tsuid'] for x in result]
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        finally:
            # clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=True)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_degraded_bad_arguments(self):
        """
        Check behavior when bad arguments provided

        case : DEGRADED
        """

        tsuid, fid = _init_nominal()
        ds_name = "DS_Test_Cut_Dataset"

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[tsuid])

        with self.assertRaises(ValueError):
            # CASE : no dataset name provided
            ds_name = None
            start_cut = int(1e12 + 3000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            # noinspection PyTypeChecker
            dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)

        with self.assertRaises(ValueError):
            # CASE : no start date provided
            ds_name = "DS_Test_Cut_Dataset"
            start_cut = None
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            # noinspection PyTypeChecker
            dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)

        with self.assertRaises(ValueError):
            # CASE : no end date nor nb points provided
            ds_name = "DS_Test_Cut_Dataset"
            start_cut = int(1e12 + 3000)
            end_cut = None
            nb_points_cut = None
            dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)

        with self.assertRaises(ValueError):
            # CASE : end date and number of points provided together
            ds_name = "DS_Test_Cut_Dataset"
            start_cut = int(1e12 + 3000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = 18
            dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)

        with self.assertRaises(ValueError):
            # CASE : end date and start date are equal
            ds_name = "DS_Test_Cut_Dataset"
            start_cut = int(1e12 + 44000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut)

        # clean up
        IkatsApi.ds.delete(ds_name, True)
