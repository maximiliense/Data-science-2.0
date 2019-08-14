import smtplib
from email.message import EmailMessage

from engine.parameters import special_parameters
from engine.logging.logs import print_errors


def send_email(subject, content):
    if special_parameters.to_mail != '':

        server = smtplib.SMTP('smtp.lirmm.fr', 25)

        msg = EmailMessage()
        c = 'Bonjour ' + special_parameters.to_name + ', \n\nVoici les resultats de ton experience : \n\n' + content
        c += '\n\nA+'
        msg.set_content(c)
        msg['Subject'] = subject
        msg['From'] = 'species_distribution_modelling@lirmm.fr'
        msg['To'] = special_parameters.to_mail

        server.send_message(msg)
        server.quit()
    else:
        print_errors('[Mail] mail not set...')
