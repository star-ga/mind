// Copyright STARGA Inc. Reference infrastructure only, not MIND implementation.
// Include the actual weak shim so its public capacity helper is checked too.
#include "../../../../runtime-support/mind_intrinsics.c"
#include <assert.h>

static int64_t header(int64_t base, int64_t len, int64_t cap) {
    int64_t rec = __mind_alloc(24);
    __mind_store_i64(rec, base);
    __mind_store_i64(rec+8, len);
    __mind_store_i64(rec+16, cap);
    return rec;
}
static void release(int64_t rec) {
    __mind_free(__mind_load_i64(rec));
    __mind_free(rec);
}
int main(void) {
    static const char literal[] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const int lengths[] = {0, 1, 3, 15, 16, 17, 31, 32};
    const int capacities[] = {16,16,16,16,32,32,32,64};
    for (unsigned k=0; k<sizeof(lengths)/sizeof(lengths[0]); ++k) {
        int n=lengths[k];
        int64_t before=header((int64_t)(uintptr_t)literal,n,0);
        int64_t after=string_push_byte(before,66);
        assert(__mind_load_i64(after+8)==n+1);
        assert(__mind_load_i64(after+16)==capacities[k]);
        assert(__mind_load_i64(after)!=(int64_t)(uintptr_t)literal);
        assert(memcmp((void *)(uintptr_t)__mind_load_i64(after),literal,(size_t)n)==0);
        assert(__mind_load_i8(__mind_load_i64(after)+n)==66);
        assert(memcmp(literal,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",32)==0);
        __mind_free(before);release(after);
    }
    // Nonnull empty borrowed address must not be read, freed or reallocated.
    int64_t empty=header(1,0,0), grown=string_push_byte(empty,255);
    assert(__mind_load_i8(__mind_load_i64(grown))==255);
    __mind_free(empty);release(grown);
    // Interior writable borrowed view: whole parent, including append location,
    // remains unchanged. The append must not free an interior pointer.
    unsigned char parent[80];memset(parent,97,sizeof(parent));
    int64_t view=header((int64_t)(uintptr_t)(parent+8),32,0);
    int64_t copy=string_push_byte(view,66);
    for (unsigned i=0;i<sizeof(parent);++i) assert(parent[i]==97);
    assert(__mind_load_i64(copy+16)==64);
    __mind_free(view);release(copy);
    // Spare capacity is reused; a single-byte store cannot damage the suffix.
    int64_t a=__mind_alloc(16);memset((void *)(uintptr_t)a,97,16);
    int64_t owned=header(a,3,8), spare=string_push_byte(owned,-1);
    assert(__mind_load_i64(spare)==a);
    assert(__mind_load_i8(a+3)==255);
    for(int i=4;i<16;i++) assert(__mind_load_i8(a+i)==97);
    __mind_free(owned);release(spare);
    // Owned realloc may move; do not inspect or free the old backing alias.
    a=__mind_alloc(32);memset((void *)(uintptr_t)a,97,32);
    owned=header(a,32,32);grown=string_push_byte(owned,66);
    assert(__mind_load_i64(grown+16)==64);
    for(int i=0;i<32;i++) assert(__mind_load_i8(__mind_load_i64(grown)+i)==97);
    assert(__mind_load_i8(__mind_load_i64(grown)+32)==66);
    __mind_free(owned);release(grown);
    assert(string_push_capacity(-1,0)==0);
    assert(string_push_capacity(0,-1)==0);
    assert(string_push_capacity(3,2)==0);
    assert(string_push_capacity(INT64_MAX,0)==0);
    assert(string_push_capacity(INT64_MAX-1,0)==INT64_MAX);
    assert(string_push_capacity(INT64_MAX-1,INT64_MAX-1)==INT64_MAX);
    assert(string_push_capacity(INT64_MAX/2,INT64_MAX/2)==INT64_MAX-1);
    assert(string_push_capacity(INT64_MAX/2+1,INT64_MAX/2+1)==INT64_MAX/2+2);
    puts("PASS reference borrowed, owned, canary and capacity controls");
    return 0;
}
