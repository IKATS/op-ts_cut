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

from ts_cut.ds_cut import dataset_cut
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


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID and funcId
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_Downsampling_%s" % ts_id

    if ts_id == 1:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3000, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7000, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 42.0],
            [1e12 + 10000, 8.0],
            [1e12 + 11000, 8.0],
            [1e12 + 12000, 8.0],
            [1e12 + 13000, 8.0]
        ]
    elif ts_id == 2:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3600, 8.0],
            [1e12 + 4000, -15.0],
            [1e12 + 5000, 2.0],
            [1e12 + 6000, 6.0],
            [1e12 + 7200, 3.0],
            [1e12 + 8000, 2.0],
            [1e12 + 9000, 5.0],
            [1e12 + 13000, 10.0]
        ]
    elif ts_id == 3:
        ts_content = [
            [1e12, 5.0],
            [1e12 + 1000, 6.0],
            [1e12 + 2000, 6.0],
            [1e12 + 3000, 8.0],
            [1e12 + 100000, 5.0],
            [1e12 + 101000, 9.0],
            [1e12 + 102000, 5.0],
            [1e12 + 103000, 10.0]
        ]
    elif ts_id == 4:
        ts_content = [
            [1e12, 42.0]
        ]
    else:
        raise NotImplementedError

    # Remove former potential story having this name
    try:
        tsuid = IkatsApi.fid.tsuid(fid=fid)
        IkatsApi.ts.delete(tsuid=tsuid, no_exception=True)
    except ValueError:
        # No TS to delete
        pass

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'], "funcId": fid}


class TestDsCut(unittest.TestCase):
    """
    Test of Downsampling computation
    """

    def _check_results(self, ts_list, result, expected_data):
        """
        Check the results of the downsampling and compare it to the expected data

        :param ts_list: list of duet tsuid/funcId to match input to output
        :param result: raw result of the operator
        :param expected_data: expected data to be used as comparison reference

        :type ts_list: list of dict
        :type result: dict
        :type expected_data: dict
        """

        # Check number of results is the same
        self.assertEqual(len(ts_list), len(result))

        # Check data content
        for index, ts_item in enumerate(ts_list):
            original_tsuid = ts_item["tsuid"]
            obtained_tsuid = result[original_tsuid]["tsuid"]
            obtained_data = IkatsApi.ts.read([obtained_tsuid])[0]

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

    @staticmethod
    def _cleanup_ts(obtained_result=None, ts_list=None):
        """
        Cleanup the time series used as inputs + resulting time series.

        :param obtained_result: raw results obtained by algorithm
        :type obtained_result: dict
        """
        if obtained_result is not None:
            for original_ts in obtained_result:
                IkatsApi.ts.delete(tsuid=obtained_result[original_ts]['tsuid'], no_exception=True)
                IkatsApi.ts.delete(tsuid=original_ts, no_exception=True)
        if ts_list is not None:
            for ts_item in ts_list:
                IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_nominal(self):
        """
        Compute the downsampling on a single time series without any constraint
        Check the time series is processed
        """

        # Prepare inputs
        ts_list = [gen_ts(1)]

        # Prepare expected output
        expected_results = {
            ts_list[0]['tsuid']: [
                [1e12, 6.0],
                [1e12 + 2000, 8.0],
                [1e12 + 4000, 2.0],
                [1e12 + 6000, 6.0],
                [1e12 + 8000, 42.0],
                [1e12 + 10000, 8.0],
                [1e12 + 12000, 8.0]
            ]
        }

        result = None
        try:
            # Call algorithm
            result = dataset_cut(ds_name="Portfolio", start=start_cut, nb_points=nb_points_cut)

            # Check the results
            self._check_results(ts_list, result, expected_results)

        finally:
            # Cleanup
            self._cleanup_ts(result)

    # No mock possible due to Spark
    def test_cut_ds_nb_points(self):
        """
        Test of a nominal cut function by providing number of points
        """
        nb_points_cut = 1700
        start_cut = 1436708800000

        results = dataset_cut(ds_name="WEBTraffic_MPR", start=start_cut, nb_points=nb_points_cut, nb_points_by_chunk=25000)

        self.assertEqual(len(results), 13)

        for ts in results:
            tsuid = ts['tsuid']
            data = IkatsApi.ts.read(tsuid)[0]
            self.assertEqual(len(data), nb_points_cut)
            self.assertEqual(IkatsApi.ts.nb_points(tsuid), nb_points_cut)
            for point in data:
                self.assertGreaterEqual(point[0], start_cut)

    # No mock possible due to multiprocessing
    def test_cut_ds_end_date(self):
        """
        Test of a nominal cut function by providing end date
        """
        start_cut = 1349102800000
        end_cut = 1349202800000
        results = dataset_cut(ds_name="Historical_hourly_weather", start=start_cut, end=end_cut)
        self.assertEqual(len(results), 13)

        for ts in results:
            tsuid = ts['tsuid']
            data = IkatsApi.ts.read(tsuid)[0]
            self.assertEqual(len(data), 15)
            for point in data:
                self.assertGreaterEqual(point[0], start_cut)
                self.assertLessEqual(point[0], end_cut)

    def test_cut_ds_no_pt_in_interval(self):
        """
        Test of a cut function with no point in interval
        """
        results = dataset_cut(ds_name="Portfolio", start=1449755766001, end=1449755766002)
        self.assertEqual(len(results), 0)
