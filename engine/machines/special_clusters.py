clusters = {
    'nef': ('nef',),
    'jean_zay': ('r0i', 'r1i', 'r2i', 'r3i', 'r4i', 'r5i', 'r6i', 'r7i', 'r8i', 'r9i'),
    'gpu4': ('gpu4',),
    'gpu2': ('gpu2',)
}

batch_clusters = ('nef', 'jean-zay')


def check_interactive_cluster(machine):
    return machine not in batch_clusters
