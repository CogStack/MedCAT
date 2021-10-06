from medcat.cli.config import get_auth_environment_vars
import os, sys

if os.name == "nt":
    import wexpect
    process = wexpect.spawn("python", [ "-m", "unittest", "package_tests"], encoding="utf-8", maxread=100000, timeout=36000)
else:
    import pexpect
    process = pexpect.spawn("python", [ "-m", "unittest", "package_tests"], encoding="utf-8", logfile="package_test_log.txt", maxread=100000, timeout=36000)

try:
    process.logfile_send = open("package_test_log.txt", 'w', encoding='utf-8')
    process.expect([".*password.*", wexpect.EOF, wexpect.TIMEOUT], timeout=36000, searchwindowsize=-1)
    process.sendline(get_auth_environment_vars()["storage_server_password"])

    print("=======================================================")
    print(process.before)
    print(process.after)
    print("All done.")
    
    assert 'ok' in process.readline() 

except Exception as e:
    print(repr(e))