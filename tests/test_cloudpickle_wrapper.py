import os
import sys
import pytest
from tempfile import mkstemp


from .utils import check_subprocess_call


class TestCloudpickleWrapper:

    def test_serialization_function_from_main(self):
        # check that the init_main_module parameter works properly
        # when using -c option, we don't need the safeguard if __name__ ..
        # and thus test LokyProcess without the extra argument. For running
        # a script, it is necessary to use init_main_module=False.
        code = """if True:
            from loky import get_reusable_executor

            def test_func(x):
                pass

            e = get_reusable_executor()
            e.submit(test_func, 42).result()
            print("ok")
        """
        cmd = [sys.executable]
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            cmd += [filename]
            check_subprocess_call(cmd, stdout_regex=r'ok', timeout=10)

            # Makes sure that if LOKY_PICKLER is set to default pickle, the
            # tasks are not wrapped with cloudpickle and it is not possible
            # using functions from the main module.
            with pytest.raises(ValueError, match=r'A task has failed to un-s'):
                check_subprocess_call(cmd, timeout=10,
                                      env={'LOKY_PICKLER': 'pickle'})
        finally:
            os.unlink(filename)

    def test_serialization_class_from_main(self):
        # check that the init_main_module parameter works properly
        # when using -c option, we don't need the safeguard if __name__ ..
        # and thus test LokyProcess without the extra argument. For running
        # a script, it is necessary to use init_main_module=False.
        code = """if True:
            from loky import get_reusable_executor

            class Test:
                def __init__(self, x=42):
                    self.x = x

                def test_func(self, x):
                    return 42

            def pass_all(*args, **kwargs):
                pass

            e = get_reusable_executor()
            e.submit(pass_all, Test()).result()
            e.submit(pass_all, x=Test()).result()
            e.submit(pass_all, 1, 2, a=0, x=Test()).result()
            assert e.submit(Test().test_func, 0).result() == 42
            print("ok")
        """
        cmd = [sys.executable]
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            cmd += [filename]
            check_subprocess_call(cmd, stdout_regex=r'ok', timeout=10)
        finally:
            os.unlink(filename)

    def test_cloudpickle_flag_wrapper(self):
        # check that the init_main_module parameter works properly
        # when using -c option, we don't need the safeguard if __name__ ..
        # and thus test LokyProcess without the extra argument. For running
        # a script, it is necessary to use init_main_module=False.
        code = """if True:
            import pytest
            from loky import get_reusable_executor
            from loky.cloudpickle_wrapper import wrap_non_picklable_objects

            @wrap_non_picklable_objects
            def test_func(x):
                return x

            @wrap_non_picklable_objects
            class Test:
                def __init__(self):
                    self.x = 42

                def return_func(self):
                    return self.x

            test_obj = Test()
            # Make sure the function and object behave correctly
            assert test_obj.x == 42
            assert test_func(42) == 42
            assert test_obj.return_func() == 42
            assert test_func(test_obj.return_func)() == 42

            # Make sure the wrapper do not make the object callable
            with pytest.raises(TypeError,
                              match="'Test' object is not callable"):
                test_obj()
            
            assert not callable(test_obj)

            # Make sure it is picklable even when the executor does not rely on
            # cloudpickle.
            e = get_reusable_executor()
            result_obj = e.submit(test_func, 42).result()
            result_obj = e.submit(id, test_obj).result()
            result_obj = e.submit(test_func, test_obj).result()
            assert result_obj.return_func() == 42

            print("ok")
        """
        cmd = [sys.executable]
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            cmd += [filename]
            check_subprocess_call(cmd, stdout_regex=r'ok', timeout=10,
                                  env={'LOKY_PICKLER': 'pickle'})
        finally:
            os.unlink(filename)
