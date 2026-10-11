# Tests and benchmarks for plasmavk.h. Users never run this: they include
# plasmavk.h and are done. plasmavk.h is the only source file; edit it directly.
#
#   make test       smoke tests (ASan + UBSan)
#   make check      everything users hit: strict C dialects, ObjC (with and
#                   without <Metal/Metal.h>), VK_NO_PROTOTYPES, C++, two-TU
#                   link, iOS, then the smoke tests
#   make bench      benchmarks on PlasmaVK
#   make bench-mvk  the same benchmarks on MoltenVK through the system loader
#
# Note: nothing here passes -framework flags. The header links Metal itself;
# building without them is part of what the tests verify.

CC      := xcrun clang
CXX     := xcrun clang++
IOS_CC  := xcrun -sdk iphoneos clang -target arm64-apple-ios26.0
BUILD   := build

WARN    := -Wall -Wextra -Werror -pedantic
CFLAGS  := -std=c99 $(WARN) -g
SAN     := -fsanitize=address,undefined -fno-omit-frame-pointer

SMOKE   := $(patsubst tests/smoke/%.c,%,$(wildcard tests/smoke/*.c))
BENCHES := $(patsubst bench/%.c,%,$(wildcard bench/*.c))

# Homebrew's loader and MoltenVK, for comparison runs.
BREW       := $(shell brew --prefix 2>/dev/null)
LOADER_LIB := $(BREW)/lib
MVK_ICD    := $(BREW)/etc/vulkan/icd.d/MoltenVK_icd.json

.PHONY: all test check bench bench-mvk clean
.PHONY: conformance conformance-mvk conformance-icd

all: test

test:
	@mkdir -p $(BUILD)/smoke
	@set -e; for t in $(SMOKE); do \
		$(CC) $(CFLAGS) $(SAN) tests/smoke/$$t.c -o $(BUILD)/smoke/$$t; \
		echo "== $$t"; ./$(BUILD)/smoke/$$t; \
	done

check:
	@mkdir -p $(BUILD)/check
	@set -e; for std in c99 c11 gnu17; do \
		echo "== implementation, -std=$$std"; \
		$(CC) -std=$$std $(WARN) -c tests/compile/impl.c -o $(BUILD)/check/impl-$$std.o; \
	done
	@echo "== implementation, Objective-C without ARC"
	@$(CC) -x objective-c -fno-objc-arc $(WARN) -c tests/compile/impl.c -o $(BUILD)/check/impl-objc.o
	@echo "== implementation, Objective-C file that also imports <Metal/Metal.h>"
	@$(CC) -x objective-c -fno-objc-arc $(WARN) -c tests/compile/impl_with_metal.m -o $(BUILD)/check/impl-with-metal.o
	@echo "== implementation, VK_NO_PROTOTYPES"
	@$(CC) -std=c99 $(WARN) -DVK_NO_PROTOTYPES -c tests/compile/impl.c -o $(BUILD)/check/impl-noproto.o
	@echo "== declarations from C++"
	@$(CXX) -std=c++17 $(WARN) -c tests/compile/user.cpp -o $(BUILD)/check/user-cpp.o
	@echo "== two translation units, no link flags"
	@$(CC) -std=c99 $(WARN) tests/compile/impl.c tests/compile/user.c -o $(BUILD)/check/two-tu
	@./$(BUILD)/check/two-tu
	@echo "== iOS (arm64, iOS 26)"
	@$(IOS_CC) -std=c99 $(WARN) tests/compile/impl.c tests/compile/user.c -o $(BUILD)/check/ios
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory conformance

# Every benchmark runs even if an earlier one reports an error (wrong output);
# the target fails at the end if any did. See bench/README.md.
bench:
	@mkdir -p $(BUILD)/bench
	@status=0; for b in $(BENCHES); do \
		$(CC) -std=c99 $(WARN) -O2 bench/$$b.c -o $(BUILD)/bench/$$b || exit 1; \
		./$(BUILD)/bench/$$b || status=1; \
	done; exit $$status

bench-mvk:
	@mkdir -p $(BUILD)/bench-loader
	@status=0; for b in $(BENCHES); do \
		$(CC) -std=c99 $(WARN) -O2 -DBENCH_USE_LOADER bench/$$b.c \
			-L$(LOADER_LIB) -lvulkan -Wl,-rpath,$(LOADER_LIB) -o $(BUILD)/bench-loader/$$b || exit 1; \
		VK_DRIVER_FILES=$(MVK_ICD) BENCH_IMPL=moltenvk ./$(BUILD)/bench-loader/$$b || status=1; \
	done; exit $$status

# Conformance tests (tests/conformance/README.md). The same sources build
# twice: black-box against PlasmaVK (its implementation linked from a separate
# translation unit), and against the system loader for any driver.
#
#   make conformance       on PlasmaVK, against the baseline (run by make check)
#   make conformance-mvk   on MoltenVK through the system loader
#   make conformance-icd ICD=/path/to/driver_icd.json   on any driver
#   CONF_ARGS='pattern -v' selects tests and passes options to the runner
CONF_DIR      := tests/conformance
CONF_SRC      := $(wildcard $(CONF_DIR)/*.c)
CONF_BASELINE := $(CONF_DIR)/baseline.txt
CONF_ARGS     ?=
CONF_SAN      := $(SAN) -fno-sanitize-recover=all # a sanitizer finding fails its test

$(BUILD)/conformance/impl.o: tests/compile/impl.c plasmavk.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CONF_SAN) -c tests/compile/impl.c -o $@

$(BUILD)/conformance/ct-plasmavk: $(CONF_SRC) $(CONF_DIR)/ct.h plasmavk.h $(BUILD)/conformance/impl.o
	$(CC) $(CFLAGS) $(CONF_SAN) $(CONF_SRC) $(BUILD)/conformance/impl.o -o $@

$(BUILD)/conformance/ct-loader: $(CONF_SRC) $(CONF_DIR)/ct.h plasmavk.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CONF_SRC) -L$(LOADER_LIB) -lvulkan -Wl,-rpath,$(LOADER_LIB) -o $@

conformance: $(BUILD)/conformance/ct-plasmavk
	./$< --baseline $(CONF_BASELINE) $(CONF_ARGS)

conformance-mvk: $(BUILD)/conformance/ct-loader
	VK_DRIVER_FILES=$(MVK_ICD) ./$< $(CONF_ARGS)

conformance-icd: $(BUILD)/conformance/ct-loader
	@test -n "$(ICD)" || { echo "usage: make conformance-icd ICD=/path/to/driver_icd.json"; exit 2; }
	VK_DRIVER_FILES=$(ICD) ./$< $(CONF_ARGS)

clean:
	rm -rf $(BUILD)
