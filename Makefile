# PlasmaVK development build. Users never run this: they include plasmavk.h.
#
#   make            regenerate src/gen/ and the single header plasmavk.h
#   make test       smoke tests against src/ (fast, errors point at src files)
#   make check      everything users hit: strict dialects, C++, iOS, two-TU
#                   link, smoke tests against the amalgamated plasmavk.h
#   make bench      benchmarks on PlasmaVK
#   make bench-mvk  the same benchmarks on MoltenVK through the system loader
#
# Note: nothing here passes -framework flags. The header links Metal itself;
# building without them is part of what the tests verify.

CC      := xcrun clang
CXX     := xcrun clang++
IOS_CC  := xcrun -sdk iphoneos clang -target arm64-apple-ios26.0
VKH     := external/vulkan-headers
BUILD   := build
TOOLS   := $(BUILD)/tools

WARN    := -Wall -Wextra -Werror -pedantic
CFLAGS  := -std=c99 $(WARN) -g
SAN     := -fsanitize=address,undefined -fno-omit-frame-pointer
DEVINC  := -Isrc -I$(VKH)/include

SRC_FILES := $(wildcard src/*.h src/*.c) src/pvk_extensions.txt
LICENSES  := LICENSE $(VKH)/LICENSE-MIT.txt
SMOKE     := $(patsubst tests/smoke/%.c,%,$(wildcard tests/smoke/*.c))
BENCHES   := $(patsubst bench/%.c,%,$(wildcard bench/*.c))

# Homebrew's loader and MoltenVK, for comparison runs.
BREW       := $(shell brew --prefix 2>/dev/null)
LOADER_LIB := $(BREW)/lib
MVK_ICD    := $(BREW)/etc/vulkan/icd.d/MoltenVK_icd.json

.PHONY: all gen test check ci bench bench-mvk clean

all: plasmavk.h

gen: $(BUILD)/gen.stamp

# Development tools are C99 too, built on demand.
$(TOOLS)/%: tools/%.c tools/tool.h tools/xml.h
	@mkdir -p $(TOOLS)
	$(CC) -std=c99 $(WARN) -O2 $< -o $@

$(BUILD)/gen.stamp: $(TOOLS)/gen_vk $(VKH)/registry/vk.xml $(SRC_FILES)
	$(TOOLS)/gen_vk --registry $(VKH)/registry/vk.xml \
		--extensions src/pvk_extensions.txt --src src --out src/gen
	@touch $@

plasmavk.h: $(BUILD)/gen.stamp $(TOOLS)/amalgamate $(LICENSES)
	$(TOOLS)/amalgamate src/plasmavk.h -I $(VKH)/include \
		$(addprefix --license ,$(LICENSES)) -o $@
	@touch $@

# ---- Smoke tests (dev: compiled straight from src/) ----

test: gen
	@mkdir -p $(BUILD)/smoke
	@set -e; for t in $(SMOKE); do \
		$(CC) $(CFLAGS) $(SAN) $(DEVINC) tests/smoke/$$t.c -o $(BUILD)/smoke/$$t; \
		echo "== $$t"; ./$(BUILD)/smoke/$$t; \
	done

# ---- What users hit, against the amalgamated header ----

check: plasmavk.h
	@mkdir -p $(BUILD)/check
	@set -e; for std in c99 c11 gnu17; do \
		echo "== implementation, -std=$$std"; \
		$(CC) -std=$$std $(WARN) -I. -c tests/compile/impl.c -o $(BUILD)/check/impl-$$std.o; \
	done
	@echo "== implementation, Objective-C without ARC"
	@$(CC) -x objective-c -fno-objc-arc $(WARN) -I. -c tests/compile/impl.c -o $(BUILD)/check/impl-objc.o
	@echo "== implementation, VK_NO_PROTOTYPES"
	@$(CC) -std=c99 $(WARN) -DVK_NO_PROTOTYPES -I. -c tests/compile/impl.c -o $(BUILD)/check/impl-noproto.o
	@echo "== declarations from C++"
	@$(CXX) -std=c++17 $(WARN) -I. -c tests/compile/user.cpp -o $(BUILD)/check/user-cpp.o
	@echo "== two translation units, no link flags"
	@$(CC) -std=c99 $(WARN) -I. tests/compile/impl.c tests/compile/user.c -o $(BUILD)/check/two-tu
	@./$(BUILD)/check/two-tu
	@echo "== iOS (arm64, iOS 26)"
	@$(IOS_CC) -std=c99 $(WARN) -I. tests/compile/impl.c tests/compile/user.c -o $(BUILD)/check/ios
	@set -e; for t in $(SMOKE); do \
		$(CC) $(CFLAGS) $(SAN) -I. tests/smoke/$$t.c -o $(BUILD)/check/$$t; \
		echo "== $$t (amalgamated)"; ./$(BUILD)/check/$$t; \
	done

# CI: everything in check, plus the committed generated files must be current.
ci: check
	@git diff --exit-code --stat -- plasmavk.h src/gen || \
		(echo "plasmavk.h or src/gen is stale: run make and commit the result"; exit 1)

# ---- Benchmarks ----

bench: gen
	@mkdir -p $(BUILD)/bench
	@set -e; for b in $(BENCHES); do \
		$(CC) -std=c99 $(WARN) -O2 $(DEVINC) bench/$$b.c -o $(BUILD)/bench/$$b; \
		./$(BUILD)/bench/$$b; \
	done

bench-mvk:
	@mkdir -p $(BUILD)/bench-loader
	@set -e; for b in $(BENCHES); do \
		$(CC) -std=c99 $(WARN) -O2 -DBENCH_USE_LOADER -I$(VKH)/include bench/$$b.c \
			-L$(LOADER_LIB) -lvulkan -Wl,-rpath,$(LOADER_LIB) -o $(BUILD)/bench-loader/$$b; \
		VK_DRIVER_FILES=$(MVK_ICD) BENCH_IMPL=moltenvk ./$(BUILD)/bench-loader/$$b; \
	done

clean:
	rm -rf $(BUILD)
