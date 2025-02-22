import logging
import logging.handlers
import os
from dotenv import load_dotenv, find_dotenv

class LoggerEmail(object):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.SMTP_SERVER = os.getenv("SMTP_SERVER")
        self.SMTP_PORT = os.getenv("SMTP_PORT")
        self.FROM_EMAIL = os.getenv("GMAIL_FROM_EMAIL")
        self.TO_EMAILS = os.getenv("GMAIL_TO_EMAILS")
        self.SUBJECT = f"[AgentPRM] Error Alert"
        self.USERNAME = os.getenv("GMAIL_USERNAME")
        self.PASSWORD = os.getenv("GMAIL_PASSWORD")

        self._activate = False

        self.logger = self.setup_logger()

    def set_activate(self, activate):
        print(f"Setting LoggerEmail's activate to {activate}")
        self._activate = activate

    def setup_logger(self):
        """Configures and returns a logger with SMTPHandler."""
        logger = logging.getLogger("ErrorLogger")

        if not logger.hasHandlers():  # Prevent duplicate handlers
            logger.setLevel(logging.ERROR)  # Only send emails for ERROR and above

            # Set up SMTP handler
            smtp_handler = logging.handlers.SMTPHandler(
                mailhost=(self.SMTP_SERVER, self.SMTP_PORT),
                fromaddr=self.FROM_EMAIL,
                toaddrs=self.TO_EMAILS,
                subject=self.SUBJECT,
                credentials=(self.USERNAME, self.PASSWORD),
                secure=()
            )

            # Set the formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            smtp_handler.setFormatter(formatter)

            # Attach handler to logger
            logger.addHandler(smtp_handler)

        return logger

    def log(self, message):
        if self._activate:
            self.logger.error(message, exc_info=True)


elogger = LoggerEmail()