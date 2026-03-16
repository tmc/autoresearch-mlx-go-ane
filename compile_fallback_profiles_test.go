package mlxgoane

import "testing"

func TestParseANECompileFallbackProfiles(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    []aneCompileFallbackProfile
		wantErr bool
	}{
		{
			name: "empty",
			raw:  "",
			want: nil,
		},
		{
			name: "single profile",
			raw:  "kANEFModelMIL:<empty>",
			want: []aneCompileFallbackProfile{
				{ModelType: "kANEFModelMIL", NetPlist: ""},
			},
		},
		{
			name: "multiple profiles with trimming and dedupe",
			raw:  " kANEFModelMIL:<empty> , kANEFModelMIL:<empty> , <empty>:fallback.plist ",
			want: []aneCompileFallbackProfile{
				{ModelType: "kANEFModelMIL", NetPlist: ""},
				{ModelType: "", NetPlist: "fallback.plist"},
			},
		},
		{
			name: "dash as empty",
			raw:  "-:-",
			want: []aneCompileFallbackProfile{
				{ModelType: "", NetPlist: ""},
			},
		},
		{
			name:    "invalid entry",
			raw:     "kANEFModelMIL",
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseANECompileFallbackProfiles(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseANECompileFallbackProfiles(%q) error=nil want non-nil", tc.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseANECompileFallbackProfiles(%q): %v", tc.raw, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("len(got)=%d want=%d got=%v", len(got), len(tc.want), got)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("profile[%d]=%v want=%v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestIsInvalidMILProgramCompileError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "invalid mil", err: errString("ANECCompile() FAILED: InvalidMILProgram"), want: true},
		{name: "generic compile", err: errString("ANECCompile() FAILED"), want: false},
		{name: "other", err: errString("timeout"), want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isInvalidMILProgramCompileError(tc.err); got != tc.want {
				t.Fatalf("isInvalidMILProgramCompileError(%v)=%v want=%v", tc.err, got, tc.want)
			}
		})
	}
}

type errString string

func (e errString) Error() string { return string(e) }
