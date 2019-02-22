"""
Copyright 2018-2019 CS Systèmes d'Information

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
from ikats.core.resource.api import IkatsApi
from ikats.algo.ts_cut import cut_ds_multiprocessing, dataset_cut


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


def _init_nominal(ts_id=1, offset=0):
    data_to_cut = np.array([
        [1e12 + 1000 + offset, 3.0],
        [1e12 + 2000 + offset, 15.0],
        [1e12 + 3000 + offset, 8.0],
        [1e12 + 6000 + offset, 25.89],
        [1e12 + 8000 + offset, 3.0],
        [1e12 + 9000 + offset, 21.2],
        [1e12 + 40000 + offset, 18],
        [1e12 + 43000 + offset, 15.0],
        [1e12 + 43500 + offset, 12.0],
        [1e12 + 44000 + offset, 7.5],
        [1e12 + 52000 + offset, 35.0]])

    return gen_ts(data_to_cut, ts_id)


# noinspection PyBroadException
class TestDsCut(unittest.TestCase):
    """
    Test of temporal cut
    """

    ds_name = "DS_Test_Cut_Dataset"

    @classmethod
    def setUpClass(cls):
        cls.tsuid1, cls.fid1 = _init_nominal(1)
        cls.tsuid2, cls.fid2 = _init_nominal(2)
        cls.tsuid3, cls.fid3 = _init_nominal(3)
        cls.tsuid4, cls.fid4 = _init_nominal(4)
        cls.tsuid5, cls.fid5 = _init_nominal(5)
        try:
            IkatsApi.ds.delete(ds_name=cls.ds_name, deep=True)
        finally:
            IkatsApi.ds.create(ds_name=cls.ds_name, description="",
                               tsuid_list=[cls.tsuid1, cls.tsuid2, cls.tsuid3, cls.tsuid4, cls.tsuid5])

    @classmethod
    def tearDownClass(cls):
        # Clean up
        IkatsApi.ds.delete(ds_name=cls.ds_name, deep=True)

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

        case: NOMINAL
        """

        start_cut = int(1e12 + 3000)
        end_cut = None
        nb_points_cut = 7
        ds_name = "DS_Test_Cut_Dataset_1"
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
        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[self.tsuid1])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 1)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_nominal_end_date(self):
        """
        Compute the cut on a single time series
        between start and end date
        Check the time series is cut as expected

        case: NOMINAL
        """

        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 44000)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset_2"
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

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[self.tsuid1])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 1)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def boundary_test_with_chunks_1(self):
        """
        Compute the cut on a single time series
        between start and end date with several chunks of data
        Check the time series is cut as expected

        case: BOUNDARY
            start date not aligned with point date (1ms less than a real point)
            end date not aligned with point date (1ms more than a real point)

        """

        start_cut = int(1e12 + 2999)
        end_cut = int(1e12 + 44001)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset_3"
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

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[self.tsuid1])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut,
                                 nb_points_by_chunk=2, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 1)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def boundary_test_with_chunks_2(self):
        """
        Compute the cut on a single time series
        between start and end date with several chunks of data
        Check the time series is cut as expected

        case: BOUNDARY
            start date not aligned with point date (1ms more than a real point)
            end date not aligned with point date (1ms less than a real point)
        """

        start_cut = int(1e12 + 3001)
        end_cut = int(1e12 + 43999)
        nb_points_cut = None
        ds_name = "DS_Test_Cut_Dataset_3"
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0]
        ])

        IkatsApi.ds.create(ds_name=ds_name, description="", tsuid_list=[self.tsuid1])

        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut,
                                 nb_points_by_chunk=2, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 1)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_dataset_multi_ts(self):
        """
        Compute the cut on a multi time series dataset
        between start and end date with several chunks of data
        Check the time series are cut as expected

        case: NOMINAL
        """
        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 44000)
        nb_points_cut = None
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

        try:
            # Call algorithm
            result = dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut,
                                 nb_points_by_chunk=7, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 5)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_degraded_bad_arguments_exception_raised(self):
        """
        Check behavior when bad arguments provided

        case: DEGRADED
        """

        with self.assertRaises(ValueError):
            # CASE: no dataset name provided
            start_cut = int(1e12 + 3000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            # noinspection PyTypeChecker
            dataset_cut(ds_name=None, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: no start date provided
            start_cut = None
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            # noinspection PyTypeChecker
            dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: no end date nor nb points provided
            start_cut = int(1e12 + 3000)
            end_cut = None
            nb_points_cut = None
            dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: end date and number of points provided together
            start_cut = int(1e12 + 3000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = 18
            dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: end date and start date are equal
            start_cut = int(1e12 + 44000)
            end_cut = int(1e12 + 44000)
            nb_points_cut = None
            dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: end date lesser than start date
            start_cut = int(1e12 + 44000)
            end_cut = int(1e12 + 40000)
            nb_points_cut = None
            dataset_cut(ds_name=self.ds_name, start=start_cut, end=end_cut, nb_points=nb_points_cut, use_spark=True)

        with self.assertRaises(ValueError):
            # CASE: number of points <0
            start_cut = int(1e12 + 44000)
            nb_points = -5
            result = dataset_cut(ds_name="unknown", start=start_cut, nb_points=nb_points, use_spark=True)
            self.assertEqual(len(result), 0)

    def test_degraded_bad_arguments_without_exception(self):
        """
        Check behavior when bad arguments provided

        case: DEGRADED
        """

        # CASE: dataset does not exist
        start_cut = int(1e12 + 44000)
        end_cut = int(1e12 + 45000)
        result = dataset_cut(ds_name="unknown", start=start_cut, end=end_cut, use_spark=True)
        self.assertEqual(len(result), 0)

        # CASE: empty dataset
        start_cut = int(1e12 + 44000)
        end_cut = int(1e12 + 45000)
        IkatsApi.ds.create("empty_ds", "", [])
        try:
            result = dataset_cut(ds_name="empty_ds", start=start_cut, end=end_cut, use_spark=True)
            self.assertEqual(len(result), 0)
        finally:
            IkatsApi.ds.delete(ds_name="empty_ds", deep=False)

    def test_degraded_no_points_in_range(self):
        """
        Check behavior when bad arguments provided

        case: DEGRADED
        """

        # CASE: 1 of all TS has no points in range
        start_cut = int(1e12 + 3000)
        end_cut = int(1e12 + 9000)
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 3000, 8.0],
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2]
        ])
        tsuid, fid = _init_nominal(6, 10000)
        ds_name = "ds_test_cut_dataset"
        try:
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
        finally:
            IkatsApi.ds.create(ds_name, "", [tsuid, self.tsuid1, self.tsuid2])
        try:
            # Call algorithm
            result = dataset_cut(ds_name=ds_name, start=start_cut, end=end_cut, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 2)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            IkatsApi.ts.delete(tsuid, no_exception=True)
            IkatsApi.ds.delete(ds_name=ds_name, deep=False)
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    def test_nb_points_cut_bigger_than_ts_length(self):
        """
        Check behavior when bad arguments provided

        case: DEGRADED
              number of points bigger than time series length
        """

        start_cut = int(1e12 + 3500)
        nb_points_cut = 10
        result_tsuids = []

        expected_result = np.array([
            [1e12 + 6000, 25.89],
            [1e12 + 8000, 3.0],
            [1e12 + 9000, 21.2],
            [1e12 + 40000, 18],
            [1e12 + 43000, 15.0],
            [1e12 + 43500, 12.0],
            [1e12 + 44000, 7.5],
            [1e12 + 52000, 35.0]])
        try:
            # Call algorithm
            result = dataset_cut(ds_name=self.ds_name, start=start_cut, nb_points=nb_points_cut, use_spark=True)
            result_tsuids = [x['tsuid'] for x in result]
            self.assertEqual(len(result_tsuids), 5)
            for tsuid_res in result_tsuids:
                self.array_equality(expected_data=expected_result, tsuid_result=tsuid_res)
        except Exception:
            self.fail("Unexpected error or assert failure in tests")
        finally:
            # Clean up
            if result_tsuids:
                for tsuid_res in result_tsuids:
                    IkatsApi.ts.delete(tsuid=tsuid_res, no_exception=True)

    # No mock possible due to multiprocessing
    def test_cut_ds_MULTIPROCESSING_nb_points(self):
        """
        Test of a nominal cut function by providing number of points
        """
        results = cut_ds_multiprocessing(ds_name=self.ds_name, sd=int(1e12 + 3000), nb_points=6, save=False)

        self.assertEqual(len(results), 5)

        for i in results:
            self.assertEqual(len(i), 6)

    # No mock possible due to multiprocessing
    def test_cut_ds_MULTIPROCESSING_end_date(self):
        """
        Test of a nominal cut function by providing end date
        """
        results = cut_ds_multiprocessing(ds_name=self.ds_name, sd=int(1e12 + 3000), ed=int(1e12 + 9000), save=False)
        self.assertEqual(len(results), 5)

        for i in results:
            self.assertEqual(len(i), 4)

    def test_cut_ds_MULTIPROCESSING_no_pt_in_interval(self):
        """
        Test of a cut function with no point in interval
        """
        results = cut_ds_multiprocessing(ds_name=self.ds_name, sd=1449755766001, ed=1449755766002, save=False)
        self.assertEqual(len(results), 5)

        for i in results:
            self.assertEqual(len(i), 0)
