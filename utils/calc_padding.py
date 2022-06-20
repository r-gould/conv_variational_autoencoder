def calc_padding(inp, out, kernel_size, stride):
    pad = (out - 1)*stride + kernel_size - inp
    print(pad)
    return (pad, pad)