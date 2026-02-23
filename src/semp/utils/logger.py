try:
    from osl_ephys.utils.logger import log_or_print
except ImportError:
    import logging

    semp_logger = logging.getLogger("semp")
    semp_logger.setLevel(logging.WARNING)

    def log_or_print(msg, warning=False):
        """Execute logger.info if a semp logger has been set up, otherwise print.

        Mirrors the signature of osl_ephys.utils.logger.log_or_print.

        Parameters
        ----------
        msg : str
            Message to log/print.
        warning : bool
            Is the msg a warning? Defaults to False, which will log/print as info.
        """
        if warning:
            msg = f"WARNING: {msg}"
        if hasattr(semp_logger, "already_setup"):
            semp_logger.info(msg)
        else:
            print(msg)
