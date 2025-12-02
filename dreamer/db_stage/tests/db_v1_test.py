import unittest
import sympy as sp
from contextlib import contextmanager
import os
from ramanujantools.cmf import pFq

from db_stage.errors import ConstantAlreadyExists, ConstantDoesNotExist
from db_stage.funcs.pFq_fmt import pFq_formatter
from dreamer.db_stage.db_s.db_v1.db import DB
from utils.position import Position


@contextmanager
def safe(testcase, exc_type=Exception):
    try:
        yield
    except exc_type as e:
        testcase.fail(f"Unexpectedly raised {type(e).__name__}: {e}")


class TestDB(unittest.TestCase):
    test_db = None
    path = f'db_v1_test.db'
    json_path = f'db_v1_test.json'

    @classmethod
    def setUpClass(cls):
        try:
            os.remove(TestDB.path)
        except:
            pass
        TestDB.test_db = DB(TestDB.path)

    @classmethod
    def tearDownClass(cls):
        del TestDB.test_db

    def test_db_main(self):
        # Test insert
        with safe(self):
            TestDB.test_db.insert('pi', pFq_formatter(2, 1, sp.Rational(1, 2)))
            TestDB.test_db.insert('e', pFq_formatter(3, 1, sp.Rational(1, 2)))
        with self.assertRaises(ConstantAlreadyExists):
            TestDB.test_db.insert('pi', pFq_formatter(2, 1, sp.Rational(1, 2)))

        # Test update
        with safe(self):
            TestDB.test_db.update('pi', pFq_formatter(2, 1, sp.Rational(1, 3)), override=True)
            TestDB.test_db.update('e', pFq_formatter(3, 1, sp.Rational(1, 3)), override=True)
        with self.assertRaises(ConstantDoesNotExist):
            TestDB.test_db.update('gamma', pFq_formatter(2, 1, sp.Rational(1, 2)))

        # Test select
        with safe(self):
            self.assertEqual(TestDB.test_db.select('pi'), [(
                pfq1 := pFq(2, 1, sp.Rational(1, 3)), Position([0, 0, 0], list(pfq1.matrices.keys()))
            )])
            self.assertEqual(TestDB.test_db.select('e'), [(
                pfq2 := pFq(3, 1, sp.Rational(1, 3)), Position([0, 0, 0, 0], list(pfq2.matrices.keys()))
            )])
        with self.assertRaises(ConstantDoesNotExist):
            TestDB.test_db.select('gamma')

        # Test replace
        with safe(self):
            TestDB.test_db.replace('pi', pFq_formatter(2, 1, sp.Rational(1, 4)))
            TestDB.test_db.replace('e', pFq_formatter(3, 1, sp.Rational(1, 4)))
            self.assertEqual(TestDB.test_db.select('pi'), [(
                pfq1 := pFq(2, 1, sp.Rational(1, 4)), Position([0, 0, 0], list(pfq1.matrices.keys()))
            )])
            self.assertEqual(TestDB.test_db.select('e'), [(
                pfq2 := pFq(3, 1, sp.Rational(1, 4)), Position([0, 0, 0, 0], list(pfq2.matrices.keys()))
            )])

        # Test delete
        with safe(self):
            TestDB.test_db.delete('pi')
            self.assertEqual(self.test_db.select('pi'), [])
        with self.assertRaises(ConstantDoesNotExist):
            TestDB.test_db.delete('gamma')

        # Test clear
        with safe(self):
            TestDB.test_db.clear()
            self.assertEqual(self.test_db.select('pi'), [])
            self.assertEqual(self.test_db.select('e'), [])

        # Test from_json
        with open(self.json_path, 'w') as f:
            f.write(
                """
                {
                    "command": "update",
                    "data": [
                        {
                            "constant": "pi",
                            "data": {
                                "type": "pFq_formatter",
                                "data": {
                                    "p": 2,
                                    "q": 1,
                                    "z": "1/3",
                                    "shifts": [0, 0, 0]
                                },
                                "kwargs": {
                                    "override": "True"
                                }
                            }
                        }
                    ]
                }
                """
            )
        with safe(self):
            TestDB.test_db.from_json_obj(self.json_path)
            self.assertEqual(TestDB.test_db.select('pi'), [(
                pfq1 := pFq(2, 1, sp.Rational(1, 3)), Position([0, 0, 0], list(pfq1.matrices.keys()))
            )])
        with self.assertRaises(ConstantDoesNotExist):
            TestDB.test_db.delete('gamma')


if __name__ == "__main__":
    unittest.main()
