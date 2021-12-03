if __name__ == "__main__":
    from medcat.cli import package, download, listmodels, config
    import plac
    import sys

    commands = {
        "download": download,
        "package": package,
        "listmodels": listmodels,
        "config": config,
    }

    if len(sys.argv) == 1:
        print("\nAvailable commands : ", ", ".join(commands))
        sys.exit()

    command = sys.argv.pop(1)
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        available = "\nAvailable commands : {}".format(", ".join(commands))
        print("Unknown command: {}".format(command), available)
        sys.exit()
