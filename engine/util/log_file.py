def save_file(path, subject, content):
    with open(path, 'a') as f:
        f.write(subject)
        f.write('\n')
        f.write(content)
