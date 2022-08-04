# Test resources

Copy `ean13_5012345678900.png` to the executable folder.

## Test command
```sh
./createsamples -img ean13_5012345678900.png -num 100 -maxxangle 0 -maxyangle 0 -maxzangle 1.6 -w 75 -h 32 -vec barcode.vec
diff expected_barcode.vec barcode.vec
```