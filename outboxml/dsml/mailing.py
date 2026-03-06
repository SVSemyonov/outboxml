import os
from typing import Tuple, List, Dict
import pandas as pd
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pretty_html_table import build_table


class Mail:
    """Class for constructing and sending HTML emails with tables, images, and formatted text.

    This class provides functionality for building HTML email messages with
    support for images, tables, formatted text, and attachments. Emails are
    sent via SMTP.

    :param config: Configuration object with email settings (SMTP server, port,
        sender, login, password, receivers).
    :type config: Any

    :var body_default: Default HTML body template with styling.
    :var config: Configuration object.
    :var msg: MIMEMultipart message object.
    :var subject: Email subject line.
    :var body: HTML email body content.
    :var signature: Email signature HTML.
    :var n_photos: Number of images added to the email.
    :var host: SMTP server hostname.
    :var port: SMTP server port.
    :var sender: Sender email address.
    :var login: SMTP login username.
    :var password: SMTP password.
    :var receivers: List of recipient email addresses.

    Example::

        from outboxml.dsml.mailing import Mail
        import config

        mail = Mail(config=config)
        mail.add_email_subject("Test Email")
        mail.add_text("Hello, this is a test email.", properties=['bold'])
        mail.add_pandas_table(df, color="blue_dark")
        mail.send_mail(receiver_emails=["recipient@example.com"])
    """
    body_default = """<html><head><meta charset="utf-8"><style> p {font-family: sans-serif, 'Times New Roman', Times, serif;} font {font-family: sans-serif, 'Times New Roman', Times, serif;}</style></head><body>"""

    def __init__(self, config):
        """Initialize Mail instance.

        :param config: Configuration object with email settings.
        :type config: Any
        """
        self.config = config
        self.msg = MIMEMultipart()
        self.subject: str = ""
        self.body: str = self.body_default
        self.signature: str = """<p><strong><br />Best regards,<br />Artificial Intelligence"""
        self.n_photos: int = 0
        self.host = self.config.email_smtp_server
        self.port = self.config.email_port
        self.sender = self.config.email_sender
        self.login = self.config.email_login
        self.password = self.config.email_pass
        self.receivers = self.config.email_receivers
       # self.logo_path = os.path.dirname(__file__) + '/ai_logo.png'

    @staticmethod
    def _image_attachment(path: str, attachment_tag: str):
        w = open(path, 'rb')
        img = MIMEImage(w.read())
        w.close()
        img.add_header('Content-ID', '<{}>'.format(attachment_tag))
        return img

    def add_image(self, bytes_, size_pixel: Tuple[int, int] = (100, 100), n_line_breaks: int = 0):
        """Add an image to the email body.

        Images can only be added in bytes format. For io.BytesIO objects,
        use ``buf.getbuffer().tobytes()``.

        :param bytes_: Image data in bytes format.
        :type bytes_: bytes
        :param size_pixel: Image size as (width, height) tuple. Defaults to (100, 100).
        :type size_pixel: Tuple[int, int]
        :param n_line_breaks: Number of line breaks after the image. Defaults to 0.
        :type n_line_breaks: int
        :return: None
        :rtype: None

        Example::

            from io import BytesIO
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            buf = BytesIO()
            plt.savefig(buf, format='png')
            mail.add_image(buf.getbuffer().tobytes(), size_pixel=(500, 300))
        """
        self.body = self.body + '<img src="cid:image{}"high="{}" width="{}"/><br/>'.format(self.n_photos + 1,
                                                                                           size_pixel[1], size_pixel[0])
        img = MIMEImage(bytes_)
        img.add_header('Content-ID', '<{}>'.format('image{}'.format(self.n_photos + 1)))

        self.msg.attach(img)
        self.n_photos = self.n_photos + 1

        self.body = self.body + "<br>" * n_line_breaks

    def add_text(self, text, properties: List = [], n_line_breaks: int = 0):
        """Add formatted text to the email body.

        :param text: Text content to add.
        :type text: str
        :param properties: List of formatting options. Supported options:
            - 'bold' - Make text bold
            - 'italic' - Make text italic
            - 'size:20' - Set font size to 20 (or any number)
            - 'size:+10' - Increase font size by 10
        :type properties: List[str]
        :param n_line_breaks: Number of line breaks after text. Defaults to 0.
        :type n_line_breaks: int
        :return: None
        :rtype: None

        Example::

            mail.add_text("Important message", properties=['bold', 'size:20'])
            mail.add_text("Regular text", n_line_breaks=2)
        """

        if 'bold' in properties:
            text = "<b>" + text + "</b>"
        if 'italic' in properties:
            text = "<i>" + text + "</i>"
        if any(['size' in i for i in properties]):
            size = ([i for i in properties if 'size' in i][0]).split(':')[1]
            text = '<font size="' + size + '">' + text + '</font>'

        self.body = self.body + '<font>' + text.replace("\n", "<br>") + '</font>' + "<br>" * n_line_breaks

    def add_pandas_table(
            self, table: pd.DataFrame, color: str = "blue_dark", replace_dict: Dict = {}, params: Dict = {}
    ):
        """Add a pandas DataFrame as an HTML table to the email body.

        :param table: DataFrame to convert to HTML table.
        :type table: pd.DataFrame
        :param color: Color scheme for the table. Defaults to "blue_dark".
        :type color: str
        :param replace_dict: Dictionary for string replacements in the HTML.
            Keys are strings to find, values are replacement strings.
        :type replace_dict: Dict[str, str]
        :param params: Additional parameters to pass to the table builder.
        :type params: Dict
        :return: None
        :rtype: None

        Example::

            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            mail.add_pandas_table(df, color="green_light", replace_dict={'NaN': 'N/A'})
        """
        def replace(item, dict_):
            for i in dict_.keys():
                item = item.replace(i, dict_[i])
            return item

        self.body = self.body + replace(build_table(table, color=color, **params), replace_dict)

    def add_line_breaks(self, n_line_breaks: int = 1):
        """Add line breaks to the email body.

        :param n_line_breaks: Number of line breaks to add. Defaults to 1.
        :type n_line_breaks: int
        :return: None
        :rtype: None
        """
        self.body = self.body + "<br>" * n_line_breaks

    def add_email_subject(self, subject: str):
        """Set the email subject line.

        :param subject: Email subject text.
        :type subject: str
        :return: None
        :rtype: None
        """
        self.subject = subject

    def send_mail(self, receiver_emails, add_signature: bool = True):
        """Send the email via SMTP.

        :param receiver_emails: List of recipient email addresses.
        :type receiver_emails: List[str]
        :param add_signature: Whether to add the signature to the email body.
            Defaults to True.
        :type add_signature: bool
        :return: None
        :rtype: None

        :raises smtplib.SMTPException: If email sending fails.
        """
        if add_signature:
            self.body = self.body + ("%s" % self.signature) + '<img src="cid:ai_logo"high="100" width="100"/><br/>'
       #     self.msg.attach(self._image_attachment(path=self.logo_path, attachment_tag="ai_logo"))

        self.msg["From"] = self.sender
        self.msg["To"] = "; ".join(self.receivers)
        self.msg["Subject"] = self.subject

        self.msg.attach(MIMEText(self.body, "html"))
       # print(self.msg.as_string())
        with smtplib.SMTP(host=self.host, port=self.port) as server:
            server.starttls(context=ssl._create_unverified_context())
            server.login(self.login, self.password)
            server.sendmail(self.sender, self.receivers, self.msg.as_string())
            server.quit()

        self.msg = None
        self.msg = MIMEMultipart()
        self.body = self.body_default
